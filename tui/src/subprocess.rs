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

/// Get the path to the Python executable in the project venv.
fn python_exe() -> PathBuf {
    let root = repo_root();

    // Check multiple common venv names
    let venv_names = [".venv", "annihilation-env", "venv", "env"];

    for venv in venv_names.iter() {
        let path = root.join(venv).join("Scripts").join("python.exe");

        if path.exists() {
            return path;
        }
    }

    // Fallback if none exist (will likely crash on spawn, but we try the standard)
    root.join(".venv").join("Scripts").join("python.exe")
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

        let mut cmd = Command::new("powershell");
        let gpu_arg = if is_gpu { "--gpu" } else { "" };
        cmd.arg("-Command");
        cmd.arg(format!("if (-not (Test-Path '.venv') -and -not (Test-Path 'annihilation-env') -and -not (Test-Path 'venv') -and -not (Test-Path 'env')) {{ Write-Output 'First run detected: Creating annihilation-env virtual environment... (NOTE: Initial setup and PyTorch extraction can take 15-20 minutes)'; python -m venv annihilation-env; Write-Output 'Virtual environment created successfully.' }}; $python = if (Test-Path 'annihilation-env') {{ '.\\annihilation-env\\Scripts\\python.exe' }} elseif (Test-Path '.venv') {{ '.\\.venv\\Scripts\\python.exe' }} elseif (Test-Path 'venv') {{ '.\\venv\\Scripts\\python.exe' }} else {{ '.\\env\\Scripts\\python.exe' }}; & $python -u verify_env_windows.py {}", gpu_arg));

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
                        for text in reader.lines().flatten() {
                            let event = parser::parse_line(&text);
                            let _ = tx_out.send(SubprocessMessage::Event(event));
                        }
                    });
                }

                if let Some(stderr) = stderr {
                    let tx_err = tx.clone();
                    thread::spawn(move || {
                        let reader = BufReader::new(stderr);
                        for text in reader.lines().flatten() {
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
    pub fn spawn_gguf_convert(model_path: &str, quant_type: &str) -> Self {
        let (tx, rx) = mpsc::channel::<SubprocessMessage>();

        let root = repo_root();
        let python = python_exe();
        let sanitized = sanitize_model_name(model_path);
        let output_dir = root.join("exports");
        let _ = std::fs::create_dir_all(&output_dir);
        let output_path = output_dir.join(format!("{sanitized}-{quant_type}.gguf"));

        let mut cmd = Command::new(&python);
        cmd.arg("-u");
        cmd.arg("scripts/gguf_converter.py");
        cmd.arg("--model-path").arg(model_path);
        cmd.arg("--quant-type").arg(quant_type);
        cmd.arg("--output").arg(&output_path);

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
                            if stdin.write_all(input.as_bytes()).is_err() { break; }
                            if stdin.write_all(b"\n").is_err() { break; }
                            let _ = stdin.flush();
                        }
                    });
                    Some(stx)
                } else {
                    None
                };

                Self { rx, child: Some(child), stdin_tx }
            }
            Err(e) => {
                let _ = tx.send(SubprocessMessage::SpawnError(format!("Failed to start GGUF conversion: {}", e)));
                Self { rx, child: None, stdin_tx: None }
            }
        }
    }

    /// Spawns the python chat server script
    pub fn spawn_chat_server(model_name: &str) -> Self {
        let (tx, rx) = mpsc::channel::<SubprocessMessage>();

        let root = repo_root();
        let python = python_exe();

        let mut cmd = Command::new(&python);
        cmd.arg("-u");
        cmd.arg("scripts/chat_server.py");
        cmd.arg(model_name);

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
                            if stdin.write_all(input.as_bytes()).is_err() { break; }
                            if stdin.write_all(b"\n").is_err() { break; }
                            let _ = stdin.flush();
                        }
                    });
                    Some(stx)
                } else {
                    None
                };

                Self { rx, child: Some(child), stdin_tx }
            }
            Err(e) => {
                let _ = tx.send(SubprocessMessage::SpawnError(format!("Failed to start chat server: {}", e)));
                Self { rx, child: None, stdin_tx: None }
            }
        }
    }



    /// Spawns the python benchmark script
    pub fn spawn_benchmark(model_name: &str) -> Self {
        let (tx, rx) = mpsc::channel::<SubprocessMessage>();

        let root = repo_root();
        let python = python_exe();

        let mut cmd = Command::new(&python);
        cmd.arg("-u");
        cmd.arg("scripts/run_benchmarks.py");
        cmd.arg(model_name);

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

                Self { rx, child: Some(child), stdin_tx: None }
            }
            Err(e) => {
                let _ = tx.send(SubprocessMessage::SpawnError(format!("Failed to start benchmark: {}", e)));
                Self { rx, child: None, stdin_tx: None }
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
        if let Some(ref mut child) = self.child {
            if let Ok(Some(status)) = child.try_wait() {
                messages.push(SubprocessMessage::Exited(status.code()));
                self.child = None; // Clean up so we only emit Exited once
            }
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
