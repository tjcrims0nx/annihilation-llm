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
fn repo_root() -> PathBuf {
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
        let path = if cfg!(windows) {
            root.join(venv).join("Scripts").join("python.exe")
        } else {
            root.join(venv).join("bin").join("python")
        };
        
        if path.exists() {
            return path;
        }
    }
    
    // Fallback if none exist (will likely crash on spawn, but we try the standard)
    if cfg!(windows) {
        root.join(".venv").join("Scripts").join("python.exe")
    } else {
        root.join(".venv").join("bin").join("python")
    }
}

fn spawn_exit_watcher(child_id: u32, tx_wait: Sender<SubprocessMessage>) {
    thread::spawn(move || {
        if cfg!(windows) {
            let script = format!(
                "$p = Get-Process -Id {} -ErrorAction SilentlyContinue; \
                 if ($null -eq $p) {{ exit 255 }}; \
                 $p.WaitForExit(); exit $p.ExitCode",
                child_id
            );

            let code = Command::new("powershell")
                .args(["-NoProfile", "-Command", &script])
                .status()
                .ok()
                .and_then(|status| status.code());

            let _ = tx_wait.send(SubprocessMessage::Exited(code));
        } else {
            // Unix: poll via /proc or kill(0)
            loop {
                thread::sleep(std::time::Duration::from_millis(500));
                // Check if process is still alive with kill(pid, 0)
                let status = Command::new("kill")
                    .args(["-0", &child_id.to_string()])
                    .status();
                match status {
                    Ok(s) if !s.success() => {
                        let _ = tx_wait.send(SubprocessMessage::Exited(Some(0)));
                        break;
                    }
                    Err(_) => {
                        let _ = tx_wait.send(SubprocessMessage::Exited(None));
                        break;
                    }
                    _ => {} // still running
                }
            }
        }
    });
}

/// Messages sent from the subprocess to the UI.
#[derive(Debug)]
pub enum SubprocessMessage {
    /// A parsed event from stdout/stderr
    Event(ParsedEvent),
    /// Raw output line (for the log panel)
    OutputLine(String),
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

        let mut cmd = if cfg!(windows) {
            let mut c = Command::new("powershell");
            let gpu_arg = if is_gpu { "--gpu" } else { "" };
            c.arg("-Command");
            c.arg(format!("if (-not (Test-Path '.venv') -and -not (Test-Path 'annihilation-env') -and -not (Test-Path 'venv') -and -not (Test-Path 'env')) {{ Write-Output 'First run detected: Creating annihilation-env virtual environment...'; python -m venv annihilation-env; Write-Output 'Virtual environment created successfully.' }}; $python = if (Test-Path 'annihilation-env') {{ '.\\annihilation-env\\Scripts\\python.exe' }} elseif (Test-Path '.venv') {{ '.\\.venv\\Scripts\\python.exe' }} elseif (Test-Path 'venv') {{ '.\\venv\\Scripts\\python.exe' }} else {{ '.\\env\\Scripts\\python.exe' }}; & $python -u verify_env.py {}", gpu_arg));
            c
        } else {
            let mut c = Command::new("sh");
            let gpu_arg = if is_gpu { "--gpu" } else { "" };
            c.arg("-c");
            c.arg(format!("if [ ! -d '.venv' ] && [ ! -d 'annihilation-env' ] && [ ! -d 'venv' ] && [ ! -d 'env' ]; then echo 'First run detected: Creating annihilation-env virtual environment...'; python3 -m venv annihilation-env; fi; if [ -d 'annihilation-env' ]; then PYTHON='./annihilation-env/bin/python'; elif [ -d '.venv' ]; then PYTHON='./.venv/bin/python'; elif [ -d 'venv' ]; then PYTHON='./venv/bin/python'; else PYTHON='./env/bin/python'; fi; $PYTHON -u verify_env.py {}", gpu_arg));
            c
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
                        for text in reader.lines().flatten() {
                            let _ = tx_out.send(SubprocessMessage::OutputLine(text));
                        }
                    });
                }

                if let Some(stderr) = stderr {
                    let tx_err = tx.clone();
                    thread::spawn(move || {
                        let reader = BufReader::new(stderr);
                        for text in reader.lines().flatten() {
                            let _ = tx_err.send(SubprocessMessage::OutputLine(text));
                        }
                    });
                }

                spawn_exit_watcher(child.id(), tx.clone());

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
        cmd.arg("-c");
        cmd.arg("import sys; from annihilate.main import main; sys.argv = ['annihilate'] + sys.argv[1:]; sys.exit(main())");
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
                                    let _ = tx_out.send(SubprocessMessage::OutputLine(text));
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
                                    let _ = tx_err.send(SubprocessMessage::OutputLine(text));
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

                spawn_exit_watcher(child.id(), tx.clone());

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
                let _ = child.kill();
            }
        }
    }

    /// Check if the subprocess is still running.
    pub fn is_running(&self) -> bool {
        self.child.is_some()
    }

    /// Poll for all pending messages (non-blocking).
    pub fn poll_messages(&self) -> Vec<SubprocessMessage> {
        let mut messages = Vec::new();
        while let Ok(msg) = self.rx.try_recv() {
            messages.push(msg);
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
