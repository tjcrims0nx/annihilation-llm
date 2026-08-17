//! Subprocess manager for running the `annihilate` Python CLI.
//!
//! Spawns `annihilate` as a child process, captures stdout/stderr line-by-line
//! via threads, and sends parsed output back to the main UI thread.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use crate::parser::{self, ParsedEvent};

/// Discover the repo root by walking up from the running binary.
/// The binary lives at `<repo>/tui/target/{debug|release}/annihilate`,
/// so the repo root is 3 levels up. Falls back to the current working directory.
pub fn repo_root() -> PathBuf {
    let mut current = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    loop {
        if current.join("pyproject.toml").exists() {
            return current;
        }
        if !current.pop() {
            break;
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// Python's executable path inside a venv layout: `Scripts\python.exe` on
/// Windows, `bin/python` everywhere else. The layout is what differs between
/// platforms, not the venv names, so one helper keeps every probe and every
/// fallback consistent.
#[cfg(windows)]
fn venv_python_rel(venv: &str) -> PathBuf {
    PathBuf::from(venv).join("Scripts").join("python.exe")
}

/// Python's executable path inside a venv layout: `Scripts\python.exe` on
/// Windows, `bin/python` everywhere else. The layout is what differs between
/// platforms, not the venv names, so one helper keeps every probe and every
/// fallback consistent.
#[cfg(not(windows))]
fn venv_python_rel(venv: &str) -> PathBuf {
    PathBuf::from(venv).join("bin").join("python")
}

/// The venv names the TUI recognises, in priority order. Both directories may
/// exist (uv creates `.venv` by default), so the order matters.
const VENV_NAMES: [&str; 4] = ["annihilation-env", ".venv", "venv", "env"];

/// Get the path to the Python executable in the project venv.
fn python_exe() -> PathBuf {
    let root = repo_root();

    // Check multiple common venv names — annihilation-env first, matching the
    // priority in spawn_setup(). Both directories may exist (uv creates .venv
    // by default), so the order matters.
    for venv in VENV_NAMES.iter() {
        let path = root.join(venv_python_rel(venv));

        if path.exists() {
            return path;
        }
    }

    // Fallback if none exist (will likely crash on spawn, but we try the standard)
    root.join(venv_python_rel(".venv"))
}

/// The Python command every spawned process resolves through PATH. On Windows
/// the py launcher and python.org installers both provide `python`; on Linux
/// and macOS the distribution package is usually `python3`, with plain
/// `python` present on some setups. Returning `None` means "no python found
/// on PATH" rather than a different command.
fn python_command() -> Option<&'static str> {
    if which("python") {
        return Some("python");
    }

    #[cfg(not(windows))]
    if which("python3") {
        return Some("python3");
    }

    None
}

/// Whether a command is available through PATH resolution.
fn which(cmd: &str) -> bool {
    Command::new(cmd)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Turn a model reference into the checkpoint filename stem used by the Python side.
///
/// This must match `checkpoint_name_for_model` in `src/annihilate/utils.py`
/// exactly: every character that is not alphanumeric, `_` or `-` becomes `--`.
/// Note that this mangles dots, so `meta-llama/Llama-3.1-8B` becomes
/// `meta-llama--Llama-3--1-8B`. A near-miss here (e.g. only replacing path
/// separators) makes the TUI look for checkpoints that will never exist.
///
/// The result is inherently filesystem-safe, since every character illegal in
/// a Windows filename is among those replaced.
pub fn checkpoint_name(model: &str) -> String {
    let mut out = String::with_capacity(model.len());
    for ch in model.chars() {
        if ch.is_alphanumeric() || ch == '_' || ch == '-' {
            out.push(ch);
        } else {
            out.push_str("--");
        }
    }
    out
}

/// Turn a model reference into a filesystem-safe stem for export filenames.
///
/// Unlike [`checkpoint_name`] this preserves dots, so an exported GGUF keeps a
/// readable name (`meta-llama--Llama-3.1-8B-Q4_K_M.gguf`). Use it only for
/// names the TUI itself owns — never for locating a checkpoint.
pub fn sanitize_model_name(model: &str) -> String {
    let replaced = model.replace(['/', '\\'], "--");
    let mut out = String::with_capacity(replaced.len());
    for ch in replaced.chars() {
        match ch {
            ':' | '*' | '?' | '"' | '<' | '>' | '|' => out.push('-'),
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    let trimmed = out.trim_matches(['.', ' ']).to_string();
    if trimmed.is_empty() {
        "model".to_string()
    } else {
        trimmed
    }
}

/// Path `spawn_gguf_convert` writes its export to.
///
/// Exposed separately so the UI can verify the artifact actually landed once
/// the converter exits — a zero exit code on its own is not proof of a file.
pub fn gguf_output_path(model_path: &str, quant_type: &str) -> PathBuf {
    let sanitized = sanitize_model_name(model_path);
    repo_root()
        .join("exports")
        .join(format!("{sanitized}-{quant_type}.gguf"))
}

/// Messages sent from the subprocess to the UI.
#[derive(Debug)]
pub enum SubprocessMessage {
    /// A parsed event from stdout/stderr
    Event(ParsedEvent),
    /// Process exited with code
    Exited(Option<i32>),
    /// Process failed to start
    SpawnError(String),
}

/// Manages an `annihilate` subprocess with async I/O.
pub struct SubprocessManager {
    /// Channel to receive messages from the subprocess threads
    pub rx: Receiver<SubprocessMessage>,
    /// Handle to the child process (for sending stdin / killing)
    child: Option<Child>,
    /// Sender for stdin to the child
    stdin_tx: Option<Sender<String>>,
}

impl SubprocessManager {
    /// Spawn the environment setup check.
    pub fn spawn_setup(is_gpu: bool) -> Self {
        let (tx, rx) = mpsc::channel::<SubprocessMessage>();

        let root = repo_root();
        let gpu_arg = if is_gpu { "--gpu" } else { "" };

        let mut cmd = if cfg!(windows) {
            // The setup chain on Windows stays on PowerShell so the exact
            // behaviour that shipped in 1.4.x is preserved: create the venv if
            // none exists, then run the verify script with the venv's python.
            let mut cmd = Command::new("powershell");
            cmd.arg("-Command");
            cmd.arg(format!("if (-not (Test-Path '.venv') -and -not (Test-Path 'annihilation-env') -and -not (Test-Path 'venv') -and -not (Test-Path 'env')) {{ Write-Output 'First run detected: Creating annihilation-env virtual environment... (NOTE: Initial setup and PyTorch extraction can take 15-20 minutes)'; python -m venv annihilation-env; Write-Output 'Virtual environment created successfully.' }}; $python = if (Test-Path 'annihilation-env') {{ '.\\annihilation-env\\Scripts\\python.exe' }} elseif (Test-Path '.venv') {{ '.\\.venv\\Scripts\\python.exe' }} elseif (Test-Path 'venv') {{ '.\\venv\\Scripts\\python.exe' }} else {{ '.\\env\\Scripts\\python.exe' }}; & $python -u verify_env.py {}", gpu_arg));
            cmd
        } else {
            // Linux and macOS have no PowerShell; the same chain runs natively:
            // create the venv if none exists, then run the verify script with
            // the venv's python. `python_exe()` returns whichever of the known
            // venv names exists first, or the `.venv` fallback when none do.
            let none_exist = !VENV_NAMES
                .iter()
                .any(|venv| root.join(venv_python_rel(venv)).exists());

            if none_exist {
                // Route through the channel, not stdout: ratatui owns the
                // terminal in raw mode, so anything written to stdout corrupts
                // the TUI. The PowerShell branch gets these messages into the
                // log panel via Write-Output through the piped stdout; this
                // is the same destination on POSIX.
                let _ = tx.send(SubprocessMessage::Event(parser::ParsedEvent::Status(
                    "First run detected: Creating annihilation-env virtual environment... (NOTE: Initial setup and PyTorch extraction can take 15-20 minutes)".to_string(),
                )));
                let create = Command::new(python_command().unwrap_or("python"))
                    .arg("-m")
                    .arg("venv")
                    .arg("annihilation-env")
                    .current_dir(&root)
                    .status();
                match create {
                    Ok(s) if s.success() => {
                        let _ = tx.send(SubprocessMessage::Event(parser::ParsedEvent::Status(
                            "Virtual environment created successfully.".to_string(),
                        )));
                    }
                    Ok(s) => {
                        let _ = tx.send(SubprocessMessage::SpawnError(format!(
                            "Setup error: creating annihilation-env failed with {s}"
                        )));
                        return Self {
                            rx,
                            child: None,
                            stdin_tx: None,
                        };
                    }
                    Err(e) => {
                        let _ = tx.send(SubprocessMessage::SpawnError(format!(
                            "Setup error: creating annihilation-env failed: {e}"
                        )));
                        return Self {
                            rx,
                            child: None,
                            stdin_tx: None,
                        };
                    }
                }
            }

            let mut cmd = Command::new(python_exe());
            cmd.arg("-u");
            cmd.arg("verify_env.py");
            if !gpu_arg.is_empty() {
                cmd.arg(gpu_arg);
            }
            cmd
        };

        cmd.current_dir(&root);

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        cmd.stdin(Stdio::piped());

        cmd.env("PYTHONIOENCODING", "utf-8");
        cmd.env("PYTHONUNBUFFERED", "1");
        cmd.env("FORCE_COLOR", "1");

        match cmd.spawn() {
            Ok(mut child) => {
                let stdout = child.stdout.take();
                let stderr = child.stderr.take();
                let _stdin = child.stdin.take();

                if let Some(stdout) = stdout {
                    let tx_out = tx.clone();
                    thread::spawn(move || {
                        let reader = BufReader::new(stdout);
                        // Stop on a read error rather than skipping it: `flatten()`
                        // would spin this thread forever on a pipe that keeps
                        // erroring. Every other reader here breaks; these two did not.
                        for text in reader.lines().map_while(Result::ok) {
                            let event = parser::parse_line(&text);
                            let _ = tx_out.send(SubprocessMessage::Event(event));
                        }
                    });
                }

                if let Some(stderr) = stderr {
                    let tx_err = tx.clone();
                    thread::spawn(move || {
                        let reader = BufReader::new(stderr);
                        for text in reader.lines().map_while(Result::ok) {
                            let event = parser::parse_line(&text);
                            let _ = tx_err.send(SubprocessMessage::Event(event));
                        }
                    });
                }

                Self {
                    rx,
                    child: Some(child),
                    stdin_tx: None,
                }
            }
            Err(e) => {
                let _ = tx.send(SubprocessMessage::SpawnError(format!("Setup error: {}", e)));
                Self {
                    rx,
                    child: None,
                    stdin_tx: None,
                }
            }
        }
    }

    /// Spawn `annihilate` with the given model and optional extra args.
    pub fn spawn(model: &str, extra_args: &[String]) -> Self {
        let (tx, rx) = mpsc::channel::<SubprocessMessage>();

        let root = repo_root();
        let python = python_exe();

        // Build command using the python executable directly to avoid block-buffering from pip .exe wrappers
        let mut cmd = Command::new(&python);
        cmd.arg("-u"); // Unbuffered output
        cmd.arg("-m");
        cmd.arg("annihilate");
        cmd.arg("--model").arg(model);
        for arg in extra_args {
            cmd.arg(arg);
        }

        // Set the working directory to the python project
        cmd.current_dir(&root);

        // Pipe stdout, stderr, stdin
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        cmd.stdin(Stdio::piped());

        // Set UTF-8 environment and unbuffer Python
        cmd.env("PYTHONIOENCODING", "utf-8");
        cmd.env("PYTHONUNBUFFERED", "1");
        cmd.env("PYTHONWARNINGS", "ignore");
        cmd.env("PYTHONUNBUFFERED", "1");
        // Force color so rich prints nice ANSI tags we can strip, and tqdm falls back to newline mode
        cmd.env("FORCE_COLOR", "1");
        // Automatically bypass the "Continue run" prompt if there is an interrupted run
        cmd.env("ANNIHILATE_AUTO_CONTINUE", "1");
        // Automatically exit any questionary prompts (prevents NoConsoleScreenBufferError crash if python is interrupted)
        cmd.env("ANNIHILATE_AUTO_SELECTS", "exit|exit|exit|exit");

        match cmd.spawn() {
            Ok(mut child) => {
                let stdout = child.stdout.take();
                let stderr = child.stderr.take();
                let stdin = child.stdin.take();

                // Stdout reader thread
                if let Some(stdout) = stdout {
                    let tx_out = tx.clone();
                    thread::spawn(move || {
                        let reader = BufReader::new(stdout);
                        for line in reader.lines() {
                            match line {
                                Ok(text) => {
                                    let event = parser::parse_line(&text);
                                    let _ = tx_out.send(SubprocessMessage::Event(event));
                                }
                                Err(_) => break,
                            }
                        }
                    });
                }

                // Stderr reader thread
                if let Some(stderr) = stderr {
                    let tx_err = tx.clone();
                    thread::spawn(move || {
                        let reader = BufReader::new(stderr);
                        for line in reader.lines() {
                            match line {
                                Ok(text) => {
                                    let event = parser::parse_line(&text);
                                    let _ = tx_err.send(SubprocessMessage::Event(event));
                                }
                                Err(_) => break,
                            }
                        }
                    });
                }

                // Stdin writer thread
                let stdin_tx = if let Some(mut stdin) = stdin {
                    let (stx, srx) = mpsc::channel::<String>();
                    thread::spawn(move || {
                        while let Ok(input) = srx.recv() {
                            if stdin.write_all(input.as_bytes()).is_err() {
                                break;
                            }
                            if stdin.write_all(b"\n").is_err() {
                                break;
                            }
                            let _ = stdin.flush();
                        }
                    });
                    Some(stx)
                } else {
                    None
                };

                Self {
                    rx,
                    child: Some(child),
                    stdin_tx,
                }
            }
            Err(e) => {
                let _ = tx.send(SubprocessMessage::SpawnError(format!(
                    "Failed to start annihilate: {}. Is it installed? Try: pip install annihilate-llm",
                    e
                )));
                Self {
                    rx,
                    child: None,
                    stdin_tx: None,
                }
            }
        }
    }

    /// Spawns the python script for converting a model to GGUF format
    pub fn spawn_gguf_converter(
        model_path: &str,
        quant_type: &str,
        trial_id: Option<usize>,
    ) -> Self {
        let (tx, rx) = mpsc::channel::<SubprocessMessage>();

        let root = repo_root();
        let python = python_exe();
        let output_path = gguf_output_path(model_path, quant_type);
        if let Some(parent) = output_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let mut cmd = Command::new(&python);
        cmd.arg("-u");
        cmd.arg("scripts/gguf_converter.py");
        cmd.arg("--model-path").arg(model_path);
        cmd.arg("--quant-type").arg(quant_type);
        cmd.arg("--output").arg(&output_path);

        if let Some(tid) = trial_id {
            cmd.arg("--trial").arg(tid.to_string());
        }

        cmd.current_dir(&root);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        cmd.stdin(Stdio::piped());
        cmd.env("PYTHONIOENCODING", "utf-8");
        cmd.env("PYTHONUNBUFFERED", "1");
        cmd.env("PYTHONWARNINGS", "ignore");

        match cmd.spawn() {
            Ok(mut child) => {
                let stdout = child.stdout.take();
                let stderr = child.stderr.take();
                let stdin = child.stdin.take();

                if let Some(stdout) = stdout {
                    let tx_out = tx.clone();
                    thread::spawn(move || {
                        let reader = BufReader::new(stdout);
                        for line in reader.lines() {
                            match line {
                                Ok(text) => {
                                    let event = parser::parse_line(&text);
                                    let _ = tx_out.send(SubprocessMessage::Event(event));
                                }
                                Err(_) => break,
                            }
                        }
                    });
                }

                if let Some(stderr) = stderr {
                    let tx_err = tx.clone();
                    thread::spawn(move || {
                        let reader = BufReader::new(stderr);
                        for line in reader.lines() {
                            match line {
                                Ok(text) => {
                                    let event = parser::parse_line(&text);
                                    let _ = tx_err.send(SubprocessMessage::Event(event));
                                }
                                Err(_) => break,
                            }
                        }
                    });
                }

                let stdin_tx = if let Some(mut stdin) = stdin {
                    let (stx, srx) = mpsc::channel::<String>();
                    thread::spawn(move || {
                        while let Ok(input) = srx.recv() {
                            if stdin.write_all(input.as_bytes()).is_err() {
                                break;
                            }
                            if stdin.write_all(b"\n").is_err() {
                                break;
                            }
                            let _ = stdin.flush();
                        }
                    });
                    Some(stx)
                } else {
                    None
                };

                Self {
                    rx,
                    child: Some(child),
                    stdin_tx,
                }
            }
            Err(e) => {
                let _ = tx.send(SubprocessMessage::SpawnError(format!(
                    "Failed to start GGUF conversion: {}",
                    e
                )));
                Self {
                    rx,
                    child: None,
                    stdin_tx: None,
                }
            }
        }
    }

    /// Spawns the python script for exporting a merged model
    pub fn spawn_export(checkpoint_path: &str, trial_id: usize, output_dir: &str) -> Self {
        let (tx, rx) = mpsc::channel::<SubprocessMessage>();

        let root = repo_root();
        let python = python_exe();

        let mut cmd = Command::new(&python);
        cmd.arg("-u");
        cmd.arg("-m");
        cmd.arg("annihilate.export");
        cmd.arg("--checkpoint").arg(checkpoint_path);
        cmd.arg("--trial-id").arg(trial_id.to_string());
        cmd.arg("--output").arg(output_dir);

        cmd.current_dir(&root);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        cmd.stdin(Stdio::piped());
        cmd.env("PYTHONIOENCODING", "utf-8");
        cmd.env("PYTHONUNBUFFERED", "1");
        cmd.env("PYTHONWARNINGS", "ignore");

        match cmd.spawn() {
            Ok(mut child) => {
                let stdout = child.stdout.take();
                let stderr = child.stderr.take();

                if let Some(stdout) = stdout {
                    let tx_out = tx.clone();
                    thread::spawn(move || {
                        let reader = BufReader::new(stdout);
                        for line in reader.lines() {
                            match line {
                                Ok(text) => {
                                    let event = parser::parse_line(&text);
                                    let _ = tx_out.send(SubprocessMessage::Event(event));
                                }
                                Err(_) => break,
                            }
                        }
                    });
                }

                if let Some(stderr) = stderr {
                    let tx_err = tx.clone();
                    thread::spawn(move || {
                        let reader = BufReader::new(stderr);
                        for line in reader.lines() {
                            match line {
                                Ok(text) => {
                                    let event = parser::parse_line(&text);
                                    let _ = tx_err.send(SubprocessMessage::Event(event));
                                }
                                Err(_) => break,
                            }
                        }
                    });
                }

                Self {
                    rx,
                    child: Some(child),
                    stdin_tx: None,
                }
            }
            Err(e) => {
                let _ = tx.send(SubprocessMessage::SpawnError(format!(
                    "Failed to start export: {}",
                    e
                )));
                Self {
                    rx,
                    child: None,
                    stdin_tx: None,
                }
            }
        }
    }

    /// Spawns the python script for uploading to Hugging Face Hub
    pub fn spawn_hf_upload(
        model_name: &str,
        trial_id: usize,
        repo_id: &str,
        hf_token: Option<&str>,
    ) -> Self {
        let (tx, rx) = mpsc::channel::<SubprocessMessage>();

        let root = repo_root();
        let python = python_exe();

        let mut cmd = Command::new(&python);
        cmd.arg("-u");
        cmd.arg("scripts/upload_to_hf.py");
        cmd.arg("--model-name").arg(model_name);
        cmd.arg("--trial").arg(trial_id.to_string());
        cmd.arg("--repo").arg(repo_id);

        cmd.current_dir(&root);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        cmd.stdin(Stdio::piped());
        cmd.env("PYTHONIOENCODING", "utf-8");
        cmd.env("PYTHONUNBUFFERED", "1");
        cmd.env("PYTHONWARNINGS", "ignore");

        if let Some(token) = hf_token
            && !token.is_empty()
        {
            cmd.env("HF_TOKEN", token);
        }

        match cmd.spawn() {
            Ok(mut child) => {
                let stdout = child.stdout.take();
                let stderr = child.stderr.take();

                if let Some(stdout) = stdout {
                    let tx_out = tx.clone();
                    thread::spawn(move || {
                        let reader = BufReader::new(stdout);
                        for line in reader.lines() {
                            match line {
                                Ok(text) => {
                                    let event = parser::parse_line(&text);
                                    let _ = tx_out.send(SubprocessMessage::Event(event));
                                }
                                Err(_) => break,
                            }
                        }
                    });
                }

                if let Some(stderr) = stderr {
                    let tx_err = tx.clone();
                    thread::spawn(move || {
                        let reader = BufReader::new(stderr);
                        for line in reader.lines() {
                            match line {
                                Ok(text) => {
                                    let event = parser::parse_line(&text);
                                    let _ = tx_err.send(SubprocessMessage::Event(event));
                                }
                                Err(_) => break,
                            }
                        }
                    });
                }

                Self {
                    rx,
                    child: Some(child),
                    stdin_tx: None,
                }
            }
            Err(e) => {
                let _ = tx.send(SubprocessMessage::SpawnError(format!(
                    "Failed to start HF upload: {}",
                    e
                )));
                Self {
                    rx,
                    child: None,
                    stdin_tx: None,
                }
            }
        }
    }

    /// Spawns the python chat server script
    pub fn spawn_chat_server(model_name: &str, trial_id: Option<usize>) -> Self {
        let (tx, rx) = mpsc::channel::<SubprocessMessage>();

        let root = repo_root();
        let python = python_exe();

        let mut cmd = Command::new(&python);
        cmd.arg("-u");
        cmd.arg("scripts/chat_server.py");
        cmd.arg(model_name);

        if let Some(tid) = trial_id {
            cmd.arg("--trial").arg(tid.to_string());
        }

        cmd.current_dir(&root);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        cmd.stdin(Stdio::piped());
        cmd.env("PYTHONIOENCODING", "utf-8");
        cmd.env("PYTHONUNBUFFERED", "1");
        cmd.env("PYTHONWARNINGS", "ignore");

        match cmd.spawn() {
            Ok(mut child) => {
                let stdout = child.stdout.take();
                let stderr = child.stderr.take();
                let stdin = child.stdin.take();

                if let Some(stdout) = stdout {
                    let tx_out = tx.clone();
                    thread::spawn(move || {
                        let reader = BufReader::new(stdout);
                        for line in reader.lines() {
                            match line {
                                Ok(text) => {
                                    // Let parser parse the JSON lines directly or raw text
                                    let event = parser::parse_line(&text);
                                    let _ = tx_out.send(SubprocessMessage::Event(event));
                                }
                                Err(_) => break,
                            }
                        }
                    });
                }

                if let Some(stderr) = stderr {
                    let tx_err = tx.clone();
                    thread::spawn(move || {
                        let reader = BufReader::new(stderr);
                        for line in reader.lines() {
                            match line {
                                Ok(text) => {
                                    let event = parser::parse_line(&text);
                                    let _ = tx_err.send(SubprocessMessage::Event(event));
                                }
                                Err(_) => break,
                            }
                        }
                    });
                }

                let stdin_tx = if let Some(mut stdin) = stdin {
                    let (stx, srx) = mpsc::channel::<String>();
                    thread::spawn(move || {
                        while let Ok(input) = srx.recv() {
                            if stdin.write_all(input.as_bytes()).is_err() {
                                break;
                            }
                            if stdin.write_all(b"\n").is_err() {
                                break;
                            }
                            let _ = stdin.flush();
                        }
                    });
                    Some(stx)
                } else {
                    None
                };

                Self {
                    rx,
                    child: Some(child),
                    stdin_tx,
                }
            }
            Err(e) => {
                let _ = tx.send(SubprocessMessage::SpawnError(format!(
                    "Failed to start chat server: {}",
                    e
                )));
                Self {
                    rx,
                    child: None,
                    stdin_tx: None,
                }
            }
        }
    }

    /// Spawns the python benchmark script
    pub fn spawn_benchmark(model_name: &str, trial_id: Option<usize>) -> Self {
        let (tx, rx) = mpsc::channel::<SubprocessMessage>();

        let root = repo_root();
        let python = python_exe();

        let mut cmd = Command::new(&python);
        cmd.arg("-u");
        cmd.arg("scripts/run_benchmarks.py");
        cmd.arg(model_name);

        if let Some(tid) = trial_id {
            cmd.arg("--trial").arg(tid.to_string());
        }

        cmd.current_dir(&root);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        cmd.stdin(Stdio::piped());
        cmd.env("PYTHONIOENCODING", "utf-8");
        cmd.env("PYTHONUNBUFFERED", "1");
        cmd.env("PYTHONWARNINGS", "ignore");

        match cmd.spawn() {
            Ok(mut child) => {
                let stdout = child.stdout.take();
                let stderr = child.stderr.take();
                let _stdin = child.stdin.take();

                if let Some(stdout) = stdout {
                    let tx_out = tx.clone();
                    thread::spawn(move || {
                        let reader = BufReader::new(stdout);
                        for line in reader.lines() {
                            match line {
                                Ok(text) => {
                                    let event = parser::parse_line(&text);
                                    let _ = tx_out.send(SubprocessMessage::Event(event));
                                }
                                Err(_) => break,
                            }
                        }
                    });
                }

                if let Some(stderr) = stderr {
                    let tx_err = tx.clone();
                    thread::spawn(move || {
                        let reader = BufReader::new(stderr);
                        for line in reader.lines() {
                            match line {
                                Ok(text) => {
                                    let event = parser::parse_line(&text);
                                    let _ = tx_err.send(SubprocessMessage::Event(event));
                                }
                                Err(_) => break,
                            }
                        }
                    });
                }

                Self {
                    rx,
                    child: Some(child),
                    stdin_tx: None,
                }
            }
            Err(e) => {
                let _ = tx.send(SubprocessMessage::SpawnError(format!(
                    "Failed to start benchmark: {}",
                    e
                )));
                Self {
                    rx,
                    child: None,
                    stdin_tx: None,
                }
            }
        }
    }

    /// Send input text to the subprocess stdin.
    pub fn send_input(&self, text: &str) -> bool {
        if let Some(ref tx) = self.stdin_tx {
            tx.send(text.to_string()).is_ok()
        } else {
            false
        }
    }

    /// Kill the subprocess and all its children.
    pub fn kill(&mut self) {
        if let Some(ref mut child) = self.child {
            let pid = child.id();
            if cfg!(windows) {
                // Use taskkill /T to kill the process tree (preventing python/powershell zombies)
                let _ = Command::new("taskkill")
                    .args(["/F", "/T", "/PID", &pid.to_string()])
                    .output();
            } else {
                // Kill child processes (like python running under sh, or multiprocessing workers)
                let _ = Command::new("pkill")
                    .args(["-9", "-P", &pid.to_string()])
                    .output();
                let _ = child.kill();
            }
        }
    }

    /// Check if the subprocess is still running.
    ///
    /// This actively polls the child rather than just testing whether a handle
    /// is held: between the child exiting and the next `poll_messages` call
    /// (which is what clears the handle) a handle-only check would still
    /// report the process as running.
    pub fn is_running(&mut self) -> bool {
        match self.child {
            Some(ref mut child) => match child.try_wait() {
                // Still running.
                Ok(None) => true,
                // Exited; leave the handle in place so `poll_messages` can
                // still surface the exit status exactly once.
                Ok(Some(_)) => false,
                // The status is unobtainable, so treat the child as gone.
                Err(_) => false,
            },
            None => false,
        }
    }

    /// Poll for all pending messages (non-blocking).
    pub fn poll_messages(&mut self) -> Vec<SubprocessMessage> {
        let mut messages = Vec::new();
        while let Ok(msg) = self.rx.try_recv() {
            messages.push(msg);
        }

        // Also reap the child if it exited, avoiding zombie processes,
        // and instantly surface the exit code to the UI thread.
        if let Some(ref mut child) = self.child
            && let Ok(Some(status)) = child.try_wait()
        {
            messages.push(SubprocessMessage::Exited(status.code()));
            self.child = None; // Clean up so we only emit Exited once
        }

        messages
    }
}

impl Drop for SubprocessManager {
    fn drop(&mut self) {
        self.kill();
    }
}

/// Get the repo root path (public for use by app.rs).
pub fn get_repo_root() -> PathBuf {
    repo_root()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference implementation of `checkpoint_name_for_model` from
    /// `src/annihilate/utils.py`, which decides the real filename on disk.
    fn python_checkpoint_name(model: &str) -> String {
        model
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '_' || c == '-' {
                    c.to_string()
                } else {
                    "--".to_string()
                }
            })
            .collect()
    }

    #[test]
    fn checkpoint_name_matches_python() {
        // Dotted versions are the case that regressed: the Python side maps
        // `.` to `--`, so anything that only rewrites path separators looks
        // for a file that never exists.
        for model in [
            "openbmb/MiniCPM5-1B",
            "meta-llama/Llama-3.1-8B",
            "mlx-community/Qwen3.6-27B-4bit",
            "microsoft/Phi-3.5-mini-instruct",
            "C:\\models\\local_model",
            "plain",
        ] {
            assert_eq!(
                checkpoint_name(model),
                python_checkpoint_name(model),
                "checkpoint name diverged from the Python side for {model:?}"
            );
        }
    }

    #[test]
    fn venv_python_rel_layout_matches_python() {
        // The layout Python's own `venv` module creates, so a near-miss here
        // (e.g. probing `Scripts` on Unix) would silently skip every venv.
        if cfg!(windows) {
            assert_eq!(
                venv_python_rel("annihilation-env"),
                PathBuf::from("annihilation-env")
                    .join("Scripts")
                    .join("python.exe")
            );
        } else {
            assert_eq!(
                venv_python_rel("annihilation-env"),
                PathBuf::from("annihilation-env").join("bin").join("python")
            );
        }
    }

    #[test]
    fn python_command_prefers_python_then_python3() {
        // Whichever of the two resolves first through PATH wins; the order
        // matters because a system can have both (e.g. `python` from pyenv
        // and `python3` from the distro), and the TUI must follow the same
        // resolution every spawn.
        if which("python") {
            assert_eq!(python_command(), Some("python"));
        } else if cfg!(not(windows)) && which("python3") {
            assert_eq!(python_command(), Some("python3"));
        } else {
            assert_eq!(python_command(), None);
        }
    }

    #[test]
    fn checkpoint_name_mangles_dots() {
        assert_eq!(
            checkpoint_name("meta-llama/Llama-3.1-8B"),
            "meta-llama--Llama-3--1-8B"
        );
    }

    #[test]
    fn sanitize_model_name_keeps_dots_but_drops_illegal_chars() {
        // Export filenames stay readable...
        assert_eq!(
            sanitize_model_name("meta-llama/Llama-3.1-8B"),
            "meta-llama--Llama-3.1-8B"
        );
        // ...but a drive-qualified path must not keep its colon, or the
        // resulting file cannot be created on Windows.
        assert_eq!(sanitize_model_name("C:\\models\\foo"), "C---models--foo");
        assert!(!sanitize_model_name("C:\\models\\foo").contains(':'));
    }

    #[test]
    fn sanitize_model_name_never_returns_empty() {
        assert_eq!(sanitize_model_name(""), "model");
        assert_eq!(sanitize_model_name("..."), "model");
    }
}
