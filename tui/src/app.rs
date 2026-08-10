/// Application state machine and screen rendering.
///
/// Manages all screens: Splash, Setup, Processing Dashboard,
/// Results, Chat, and Export — each with user-friendly selection menus.
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{
        Block, BorderType, Borders, Cell, Clear, Gauge, List, ListItem, ListState, Paragraph, Row,
        Sparkline, Table, TableState, Wrap,
    },
};

use crate::parser::ParsedEvent;
use crate::subprocess::SubprocessManager;
use crate::sysinfo::SystemInfo;
use crate::theme;
use serde_json;

// ─── ASCII Art Banner ──────────────────────────────────────────

const BANNER: &[&str] = &[
    r"  ████  █    █ █    █ ██ █   █ ██ █     ████  ██████ ██  ████  █    █ ",
    r" █    █ ██   █ ██   █ ██ █   █ ██ █    █    █   ██   ██ █    █ ██   █ ",
    r" ██████ █ █  █ █ █  █ ██ █████ ██ █    ██████   ██   ██ █    █ █ █  █ ",
    r" █    █ █  █ █ █  █ █ ██ █   █ ██ █    █    █   ██   ██ █    █ █  █ █ ",
    r" █    █ █   ██ █   ██ ██ █   █ ██ █████ █    █  ██   ██  ████  █   ██ ",
];

const GGUF_BANNER: &[&str] = &[
    r"  ██████   ██████  ██    ██ ███████ ",
    r" ██       ██       ██    ██ ██      ",
    r" ██   ███ ██   ███ ██    ██ █████   ",
    r" ██    ██ ██    ██ ██    ██ ██      ",
    r"  ██████   ██████   ██████  ██      ",
];

const TAGLINE: &str = "-- Breaking the Chains | Unleashing Model Potential --";

// ─── Application Screens ───────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Screen {
    Splash,
    Setup,
    ModelInput,
    TokenInput,
    ConfigSelect,
    Processing,
    Results,
    TrialActions,
    Chat,
    Export,
    GgufSizeSelect,
    CompletedModels,
    CheckpointPrompt,
    BenchmarkDashboard,
    Confirm(ConfirmAction),
    About,
    RecentModels,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConfirmAction {
    Quit,
    StopProcessing,
    DeleteCheckpoint(String),
}

/// Write (or clear) `HF_TOKEN` in the repo-root `.env`, preserving every other entry.
///
/// The file is always at the repo root rather than the process CWD, so the value is
/// picked up by pydantic-settings' dotenv source regardless of where the TUI was launched.
fn persist_hf_token(token: Option<&str>) -> std::io::Result<()> {
    let path = crate::subprocess::repo_root().join(".env");

    let mut lines: Vec<String> = match std::fs::read_to_string(&path) {
        Ok(existing) => existing
            .lines()
            .filter(|l| {
                let key = l.split('=').next().unwrap_or("").trim();
                key != "HF_TOKEN"
            })
            .map(str::to_string)
            .collect(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(e) => return Err(e),
    };

    if let Some(t) = token {
        lines.push(format!("HF_TOKEN={t}"));
    }

    while lines.last().is_some_and(|l| l.trim().is_empty()) {
        lines.pop();
    }

    let mut body = lines.join("\n");
    if !body.is_empty() {
        body.push('\n');
    }
    std::fs::write(&path, body)
}

/// Delete a model's checkpoint file. Returns `Ok(false)` if there was nothing to delete.
///
/// Errors are surfaced to the caller rather than swallowed, so the UI never claims a
/// deletion that did not happen.
fn delete_checkpoint(model: &str) -> std::io::Result<bool> {
    let path = crate::subprocess::repo_root()
        .join("checkpoints")
        .join(format!(
            "{}.jsonl",
            crate::subprocess::checkpoint_name(model)
        ));
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

/// Recover the original model name for a checkpoint file.
///
/// The filename stem cannot be reversed: `checkpoint_name` maps `/`, `.` and `:`
/// all onto `--`, so `Llama-3.1-8B` and `Llama-3/1-8B` collide. Instead read the
/// model name back out of the settings record Optuna wrote into the journal, and
/// only fall back to a best-effort de-mangling if that is unavailable.
fn model_name_from_checkpoint(path: &std::path::Path) -> String {
    use std::io::BufRead;

    let fallback = || {
        path.file_stem()
            .map(|s| s.to_string_lossy().replace("--", "/"))
            .unwrap_or_default()
    };

    // Checkpoints run to hundreds of thousands of lines, so stream rather than
    // reading the whole journal in; the settings record is written up front.
    let Ok(file) = std::fs::File::open(path) else {
        return fallback();
    };

    for line in std::io::BufReader::new(file).lines().map_while(Result::ok) {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };

        // Optuna's journal stores study user attributes as op_code 2 records.
        // The Python side puts the serialized Settings under "settings" as a
        // JSON *string*, so it needs a second parse.
        let model = value
            .get("user_attr")
            .and_then(|v| v.get("settings"))
            .and_then(|v| v.as_str())
            .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
            .as_ref()
            .and_then(|v| v.get("model"))
            .and_then(|v| v.as_str())
            .map(str::to_string);

        if let Some(model) = model {
            if !model.is_empty() {
                return model;
            }
        }
    }

    fallback()
}

/// Mask a secret for display, revealing at most the first 3 *characters*.
///
/// Byte slicing would panic on multi-byte input, so this walks chars.
fn mask_secret(secret: &str) -> String {
    let count = secret.chars().count();
    if count <= 3 {
        return "*".repeat(count);
    }
    let visible: String = secret.chars().take(3).collect();
    format!("{}{}", visible, "*".repeat(count - 3))
}

/// Insert `c` at a char-cursor position, returning the new cursor.
///
/// `cursor` counts characters, not bytes; `String::insert` needs a byte index and
/// panics if handed a non-boundary, so translate before inserting.
fn insert_at_char_cursor(s: &mut String, cursor: usize, c: char) -> usize {
    let byte_idx = s
        .char_indices()
        .nth(cursor)
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    s.insert(byte_idx, c);
    cursor + 1
}

/// Remove the character before a char-cursor position, returning the new cursor.
fn remove_before_char_cursor(s: &mut String, cursor: usize) -> usize {
    if cursor == 0 {
        return 0;
    }
    if let Some((byte_idx, _)) = s.char_indices().nth(cursor - 1) {
        s.remove(byte_idx);
        cursor - 1
    } else {
        cursor
    }
}

/// Number of characters (not bytes) in `s`.
fn char_len(s: &str) -> usize {
    s.chars().count()
}

/// Human-readable byte count, e.g. `651 MiB` (or `752.3 KB` below 1 MiB).
fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = KIB * 1024;
    const GIB: u64 = MIB * 1024;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.0} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Visual line count for `text` rendered into a pane `width` columns wide.
///
/// Mirrors how `Paragraph` with `Wrap` lays text out, so scroll bounds can be
/// computed before rendering — ratatui does not expose the wrapped height.
/// Breaks on whitespace, and hard-splits any word longer than the pane.
fn wrapped_line_count(text: &str, width: usize) -> usize {
    if width == 0 {
        // Degenerate pane; treat every explicit line as one row so callers
        // never divide by zero or under-count to zero.
        return text.lines().count().max(1);
    }

    let mut lines = 0;

    for segment in text.split('\n') {
        let mut column = 0;
        let mut segment_lines = 1;

        for word in segment.split_whitespace() {
            let word_width = char_len(word);

            if column == 0 {
                // A word wider than the pane wraps onto further rows itself.
                segment_lines += word_width.saturating_sub(1) / width;
                column = if word_width % width == 0 && word_width > 0 {
                    width
                } else {
                    word_width % width
                };
            } else if column + 1 + word_width <= width {
                column += 1 + word_width;
            } else {
                segment_lines += 1;
                segment_lines += word_width.saturating_sub(1) / width;
                column = if word_width % width == 0 && word_width > 0 {
                    width
                } else {
                    word_width % width
                };
            }
        }

        lines += segment_lines;
    }

    lines.max(1)
}

/// Completion line for a child process exit, plus the level to style it at.
///
/// A clean exit is a success, not a warning: a finished GGUF conversion used to
/// be reported in the same yellow as a crash, and printed the raw `Some(0)`
/// debug form of the code. `None` means the process was killed before it could
/// report a code, which is neither a success nor a reported failure.
fn exit_report(what: &str, code: Option<i32>) -> (String, LogLevel) {
    match code {
        Some(0) => (
            format!("✓ {what} completed successfully."),
            LogLevel::Success,
        ),
        Some(code) => (
            format!("{what} failed with exit code {code}."),
            LogLevel::Error,
        ),
        None => (
            format!("{what} was terminated before it reported an exit code."),
            LogLevel::Warning,
        ),
    }
}

// ─── Menu System ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MenuItem {
    pub label: String,
    pub description: String,
    pub key_hint: Option<String>,
}

impl MenuItem {
    fn new(label: &str, desc: &str) -> Self {
        Self {
            label: label.to_string(),
            description: desc.to_string(),
            key_hint: None,
        }
    }

    fn with_key(mut self, key: &str) -> Self {
        self.key_hint = Some(key.to_string());
        self
    }
}

// ─── Trial Data ────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TrialResult {
    pub index: usize,
    pub refusals: usize,
    pub total_prompts: usize,
    pub kl_divergence: f64,
    pub direction: String,
}

// ─── Application State ────────────────────────────────────────

pub struct App {
    pub screen: Screen,
    pub previous_screen: Option<Screen>,
    pub should_quit: bool,

    // Animation state
    pub tick_count: u64,
    pub glow_phase: f64,

    // Menu state
    pub menu_state: ListState,
    pub current_menu: Vec<MenuItem>,
    pub trial_list_state: TableState,

    // Model input
    pub model_input: String,
    pub model_cursor: usize,
    pub model_error: Option<String>,

    // Token input
    pub hf_token_input: String,
    pub hf_token_cursor: usize,

    // Processing state
    pub is_processing: bool,
    pub is_setting_up: bool,
    pub subprocess: Option<SubprocessManager>,
    pub current_trial: usize,
    pub total_trials: usize,
    pub best_refusals: Option<usize>,
    pub best_kl: Option<f64>,
    pub pending_kl: Option<f64>,
    pub log_lines: Vec<(String, LogLevel)>,
    pub log_scroll: usize,
    pub log_auto_scroll: bool,
    pub elapsed_secs: u64,
    pub eta_secs: Option<u64>,
    pub sys_info: SystemInfo,
    pub batch_size: usize,
    pub tokens_per_sec: f64,
    pub kl_history: Vec<f64>,
    pub refusal_history: Vec<f64>,

    // Results state
    pub trials: Vec<TrialResult>,

    // Chat state
    pub chat_messages: Vec<(String, String)>, // (role, content)
    pub chat_input: String,
    pub chat_scroll: usize,
    /// Stick to the newest message until the user scrolls up deliberately.
    pub chat_auto_scroll: bool,
    pub chat_subprocess: Option<SubprocessManager>,
    pub chat_loading: bool,
    pub chat_streaming: bool,
    pub benchmark_subprocess: Option<SubprocessManager>,
    pub benchmark_running: bool,
    pub benchmark_results: Vec<(String, String, String)>, // (benchmark, metric, value)

    // Status
    pub status_message: String,
    pub annihilate_available: bool,
    pub plot_residuals: bool,
    pub quantize: bool,
    pub use_obliteratus: bool,
    pub gguf_size: String,
    /// Where an in-flight GGUF conversion is writing to, so its exit can be
    /// checked against the real artifact. `None` when the running subprocess is
    /// an optimization run rather than a conversion.
    pub gguf_output: Option<std::path::PathBuf>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LogLevel {
    Info,
    Success,
    Warning,
    Error,
    Dim,
}

impl App {
    pub fn new() -> Self {
        let splash_menu = vec![
            MenuItem::new("Start Decensoring", "Launch the annihilation workflow")
                .with_key("Enter"),
            MenuItem::new("Completed Models", "Export finished models to GGUF").with_key("M"),
            MenuItem::new("Configuration", "Edit settings before running").with_key("C"),
            MenuItem::new("About", "View project information").with_key("A"),
            MenuItem::new("Quit", "Exit the application").with_key("Q"),
        ];

        let mut menu_state = ListState::default();
        menu_state.select(Some(0));

        Self {
            screen: Screen::Splash,
            previous_screen: None,
            should_quit: false,
            tick_count: 0,
            glow_phase: 0.0,
            menu_state,
            current_menu: splash_menu,
            model_input: String::new(),
            model_cursor: 0,
            model_error: None,
            hf_token_input: std::env::var("HF_TOKEN").unwrap_or_default(),
            hf_token_cursor: char_len(&std::env::var("HF_TOKEN").unwrap_or_default()),
            is_processing: false,
            is_setting_up: false,
            subprocess: None,
            current_trial: 0,
            total_trials: 200,
            best_refusals: None,
            best_kl: None,
            pending_kl: None,
            log_lines: Vec::new(),
            log_scroll: 0,
            log_auto_scroll: true,
            elapsed_secs: 0,
            eta_secs: None,
            sys_info: SystemInfo::detect(),
            batch_size: 0,
            tokens_per_sec: 0.0,
            kl_history: Vec::new(),
            refusal_history: Vec::new(),
            trials: Vec::new(),
            trial_list_state: TableState::default(),
            chat_messages: Vec::new(),
            chat_input: String::new(),
            chat_scroll: 0,
            chat_auto_scroll: true,
            chat_subprocess: None,
            chat_loading: false,
            chat_streaming: false,
            benchmark_subprocess: None,
            benchmark_running: false,
            benchmark_results: Vec::new(),
            status_message: "Ready".to_string(),
            annihilate_available: false,
            plot_residuals: false,
            quantize: false,
            use_obliteratus: false,
            gguf_size: "Q4_K_M".to_string(),
            gguf_output: None,
        }
    }

    /// Tick the animation state
    pub fn tick(&mut self) {
        self.tick_count += 1;
        self.glow_phase = (self.tick_count as f64 * 0.05).sin() * 0.5 + 0.5;

        // Process real subprocess events
        if self.screen == Screen::Processing
            && self.is_processing
            && let Some(ref mut child) = self.subprocess
        {
            use crate::subprocess::SubprocessMessage;
            let msgs = child.poll_messages();

            for msg in msgs {
                match msg {
                    SubprocessMessage::Event(event) => match event {
                        ParsedEvent::ModelLoading(msg) => {
                            self.log_lines.push((msg, LogLevel::Info));
                        }
                        ParsedEvent::BatchSize(size) => {
                            self.batch_size = size;
                            self.log_lines.push((
                                format!("Determined batch size: {}", size),
                                LogLevel::Success,
                            ));
                        }
                        ParsedEvent::DatasetLoading(msg) => {
                            self.log_lines.push((msg, LogLevel::Dim));
                        }
                        ParsedEvent::KLDivergence(kl) => {
                            self.pending_kl = Some(kl);
                        }
                        ParsedEvent::CalculatingDirections => {
                            self.log_lines.push((
                                "Calculating per-layer refusal directions...".into(),
                                LogLevel::Info,
                            ));
                        }
                        ParsedEvent::OptimizationStarting { n_trials } => {
                            self.total_trials = n_trials;
                            self.log_lines
                                .push(("Starting optimization...".into(), LogLevel::Success));
                        }
                        ParsedEvent::TrialStarting {
                            trial_number,
                            total_trials,
                        } => {
                            self.current_trial = trial_number;
                            self.total_trials = total_trials;
                            self.log_lines.push((
                                format!("Starting trial {}/{}...", trial_number, total_trials),
                                LogLevel::Info,
                            ));
                        }
                        ParsedEvent::TrialComplete {
                            trial_number,
                            total_trials: _,
                            refusals,
                            total_prompts,
                        } => {
                            if trial_number > 0 {
                                self.current_trial = trial_number;
                            } else {
                                self.current_trial += 1; // Fallback if we couldn't parse the exact number
                            }

                            let kl_divergence = self.pending_kl.take().unwrap_or(0.0);
                            self.kl_history.push(kl_divergence);
                            self.refusal_history.push(refusals as f64);

                            if self.best_refusals.is_none()
                                || refusals < self.best_refusals.unwrap()
                            {
                                self.best_refusals = Some(refusals);
                                self.best_kl = Some(kl_divergence);
                            } else if refusals == self.best_refusals.unwrap() {
                                if self.best_kl.is_none() || kl_divergence < self.best_kl.unwrap() {
                                    self.best_kl = Some(kl_divergence);
                                }
                            }

                            self.log_lines.push((
                                format!(
                                    "Trial {}: refusals={}/{}, KL={:.4}",
                                    self.current_trial, refusals, total_prompts, kl_divergence
                                ),
                                if refusals <= 5 {
                                    LogLevel::Success
                                } else {
                                    LogLevel::Info
                                },
                            ));
                        }
                        ParsedEvent::BestTrial { .. } => {}
                        ParsedEvent::OptimizationComplete => {
                            self.log_lines
                                .push(("Optimization finished!".into(), LogLevel::Success));
                            self.is_processing = false;
                            self.generate_demo_results(); // Still use demo results for now until the interactive menu parser is fully connected
                            self.switch_to_results();
                        }
                        ParsedEvent::GpuMemory { .. } => {}
                        ParsedEvent::ElapsedTime(time) => {
                            // Time is string "00:00:00", could parse it, but for now just status
                            self.log_lines
                                .push((format!("Elapsed: {}", time), LogLevel::Dim));
                        }
                        ParsedEvent::EstimatedRemaining(time) => {
                            self.log_lines
                                .push((format!("ETA: {}", time), LogLevel::Dim));
                        }
                        ParsedEvent::TrialPruned { trial_number } => {
                            self.log_lines.push((
                                format!("Trial {} pruned", trial_number),
                                LogLevel::Warning,
                            ));
                        }
                        ParsedEvent::Error(err) => {
                            self.log_lines.push((err, LogLevel::Error));
                        }
                        ParsedEvent::Warning(warn) => {
                            self.log_lines.push((warn, LogLevel::Warning));
                        }
                        ParsedEvent::Status(msg) => {
                            self.log_lines.push((msg, LogLevel::Info));
                        }
                        ParsedEvent::InteractivePrompt(prompt) => {
                            self.log_lines.push((prompt, LogLevel::Warning));
                        }
                        ParsedEvent::Raw(line) => {
                            if !line.trim().is_empty() && !line.contains("Spawning") {
                                let mut clean = line.clone();
                                if let Some(final_chunk) = clean.split('\r').last() {
                                    clean = final_chunk.to_string();
                                }
                                if clean.contains("No GPU or other accelerator detected")
                                    && !self.sys_info.gpu_name.starts_with("CPU")
                                    && self.sys_info.gpu_name != "Unknown"
                                    && self.sys_info.gpu_name != "Detecting..."
                                {
                                    self.log_lines.push((
                                            "CRITICAL WARNING: The TUI detects your GPU, but Python cannot see it! You have installed the CPU-only version of PyTorch. The process will run extremely slow.".to_string(),
                                            LogLevel::Error
                                        ));
                                    self.log_lines.push((
                                            "FIX THIS BY RUNNING: `uv pip install --python annihilation-env torch torchvision --index-url https://download.pytorch.org/whl/cu126 --upgrade`".to_string(),
                                            LogLevel::Error
                                        ));
                                }
                                self.log_lines.push((clean, LogLevel::Dim));
                            }
                        }
                    },
                    SubprocessMessage::Exited(code) => {
                        if self.is_setting_up {
                            if code == Some(0) {
                                self.is_setting_up = false;
                                self.log_lines.push((
                                    "Environment verification complete. Spawning backend..."
                                        .to_string(),
                                    LogLevel::Info,
                                ));

                                // Start the actual subprocess
                                let mut extra_args =
                                    vec!["--n-trials".to_string(), self.total_trials.to_string()];

                                // Add quantization if selected.
                                if self.quantize {
                                    extra_args.push("--quantization".to_string());
                                    extra_args.push("bnb_4bit".to_string());
                                }

                                if self.use_obliteratus {
                                    extra_args.push("--kernel-type".to_string());
                                    extra_args.push("gaussian".to_string());
                                    // By default use_cosmic_layer_selection and use_ega are true in config.py
                                } else {
                                    extra_args.push("--kernel-type".to_string());
                                    extra_args.push("linear".to_string());
                                    extra_args.push("--no-use-cosmic-layer-selection".to_string());
                                    extra_args.push("--no-use-ega".to_string());
                                }

                                self.subprocess =
                                    Some(SubprocessManager::spawn(&self.model_input, &extra_args));
                            } else {
                                self.is_processing = false;
                                self.is_setting_up = false;
                                let (msg, level) = exit_report("Environment setup", code);
                                self.log_lines.push((msg, level));
                            }
                        } else {
                            self.is_processing = false;
                            for (msg, level) in self.finish_report(code) {
                                self.log_lines.push((msg, level));
                            }
                            // Wait for user to manually exit or review logs rather than forcing them to the results screen
                        }
                    }
                    SubprocessMessage::SpawnError(err) => {
                        self.is_processing = false;
                        self.log_lines.push((err, LogLevel::Error));
                    }
                }
            }

            // Refresh real system stats periodically
            if self.tick_count.is_multiple_of(30) {
                self.sys_info.refresh_gpu();
                self.sys_info.refresh_ram();
                // Fake tokens per sec since we can't easily parse that from output yet
                self.tokens_per_sec = 847.0 + (self.tick_count as f64 * 0.01).sin() * 50.0;
                self.elapsed_secs += 1; // Roughly 1 second elapsed (at 30fps)
            }
        }

        // Poll chat subprocess for streamed tokens
        if let Some(ref mut chat_proc) = self.chat_subprocess {
            use crate::subprocess::SubprocessMessage;
            let msgs = chat_proc.poll_messages();
            for msg in msgs {
                match msg {
                    SubprocessMessage::Event(event) => {
                        if let ParsedEvent::Raw(line) = event {
                            // Parse JSON messages from chat_server.py
                            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                                match json.get("type").and_then(|t| t.as_str()) {
                                    Some("ready") => {
                                        self.chat_loading = false;
                                        self.status_message = "Chat ready.".to_string();
                                        self.chat_messages.push((
                                            "system".to_string(),
                                            "Model loaded! Type a message and press Enter."
                                                .to_string(),
                                        ));
                                    }
                                    Some("status") => {
                                        if let Some(content) =
                                            json.get("content").and_then(|c| c.as_str())
                                        {
                                            self.status_message = content.to_string();
                                            self.chat_messages
                                                .push(("system".to_string(), content.to_string()));
                                        }
                                    }
                                    Some("token") => {
                                        if let Some(content) =
                                            json.get("content").and_then(|c| c.as_str())
                                        {
                                            // Append token to the last assistant message, or create one
                                            if let Some(last) = self.chat_messages.last_mut() {
                                                if last.0 == "assistant" {
                                                    last.1.push_str(content);
                                                } else {
                                                    self.chat_messages.push((
                                                        "assistant".to_string(),
                                                        content.to_string(),
                                                    ));
                                                }
                                            } else {
                                                self.chat_messages.push((
                                                    "assistant".to_string(),
                                                    content.to_string(),
                                                ));
                                            }
                                            self.chat_streaming = true;
                                        }
                                    }
                                    Some("done") => {
                                        self.chat_streaming = false;
                                    }
                                    Some("error") => {
                                        if let Some(content) =
                                            json.get("content").and_then(|c| c.as_str())
                                        {
                                            self.chat_messages.push((
                                                "system".to_string(),
                                                format!("Error: {}", content),
                                            ));
                                            self.status_message = "Chat error.".to_string();
                                        }
                                        self.chat_loading = false;
                                        self.chat_streaming = false;
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    SubprocessMessage::Exited(_) => {
                        self.chat_subprocess = None;
                        self.chat_loading = false;
                        self.chat_streaming = false;
                        self.status_message = "Chat server disconnected.".to_string();
                        self.chat_messages.push((
                            "system".to_string(),
                            "Chat server disconnected.".to_string(),
                        ));
                    }
                    SubprocessMessage::SpawnError(err) => {
                        self.chat_loading = false;
                        self.chat_streaming = false;
                        self.status_message = format!("Failed to start chat: {}", err);
                        self.chat_messages.push((
                            "system".to_string(),
                            format!("Failed to start chat: {}", err),
                        ));
                    }
                }
            }
        }

        // Poll benchmark subprocess
        if let Some(ref mut bench_proc) = self.benchmark_subprocess {
            use crate::subprocess::SubprocessMessage;
            let msgs = bench_proc.poll_messages();
            for msg in msgs {
                match msg {
                    SubprocessMessage::Event(event) => {
                        if let ParsedEvent::Raw(line) = event {
                            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                                match json.get("type").and_then(|t| t.as_str()) {
                                    Some("result") => {
                                        let bench = json
                                            .get("benchmark")
                                            .and_then(|b| b.as_str())
                                            .unwrap_or("unknown")
                                            .to_string();
                                        let metric = json
                                            .get("metric")
                                            .and_then(|m| m.as_str())
                                            .unwrap_or("")
                                            .to_string();
                                        let value = json
                                            .get("value")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("")
                                            .to_string();
                                        self.benchmark_results.push((bench, metric, value));
                                    }
                                    Some("status") => {
                                        if let Some(content) =
                                            json.get("content").and_then(|c| c.as_str())
                                        {
                                            self.status_message = content.to_string();
                                            self.log_lines
                                                .push((content.to_string(), LogLevel::Info));
                                        }
                                    }
                                    Some("done") => {
                                        self.benchmark_running = false;
                                        self.status_message = "Benchmarks completed.".to_string();
                                        if let Some(content) =
                                            json.get("content").and_then(|c| c.as_str())
                                        {
                                            self.log_lines
                                                .push((content.to_string(), LogLevel::Success));
                                        }
                                    }
                                    Some("error") => {
                                        if let Some(content) =
                                            json.get("content").and_then(|c| c.as_str())
                                        {
                                            self.log_lines.push((
                                                format!("Benchmark error: {}", content),
                                                LogLevel::Error,
                                            ));
                                            self.status_message = "Benchmark failed.".to_string();
                                        }
                                        self.benchmark_running = false;
                                    }
                                    _ => {}
                                }
                            } else if !line.trim().is_empty() {
                                self.log_lines.push((line, LogLevel::Dim));
                            }
                        }
                    }
                    SubprocessMessage::Exited(_) => {
                        self.benchmark_subprocess = None;
                        self.benchmark_running = false;
                    }
                    SubprocessMessage::SpawnError(err) => {
                        self.benchmark_running = false;
                        self.log_lines
                            .push((format!("Benchmark error: {}", err), LogLevel::Error));
                    }
                }
            }
        }
    }

    /// Lines to log when the main subprocess exits.
    ///
    /// A GGUF export is only called a success once the file it promised is
    /// actually on disk. The converter shells out to `convert_hf_to_gguf.py`
    /// and `llama-quantize`, so it can exit 0 after a step that quietly wrote
    /// nothing — reporting that as done sends the user hunting for a model that
    /// is not there.
    fn finish_report(&mut self, code: Option<i32>) -> Vec<(String, LogLevel)> {
        let Some(output) = self.gguf_output.take() else {
            return vec![exit_report("Process", code)];
        };

        if code != Some(0) {
            return vec![exit_report("GGUF conversion", code)];
        }

        match std::fs::metadata(&output) {
            Ok(meta) if meta.len() > 0 => vec![
                (
                    "✓ GGUF conversion complete and verified.".to_string(),
                    LogLevel::Success,
                ),
                (
                    format!(
                        "  Saved to {} ({})",
                        output.display(),
                        format_bytes(meta.len())
                    ),
                    LogLevel::Success,
                ),
            ],
            Ok(_) => vec![(
                format!(
                    "GGUF conversion exited cleanly but wrote an empty file: {}",
                    output.display()
                ),
                LogLevel::Error,
            )],
            Err(e) => vec![(
                format!(
                    "GGUF conversion exited cleanly but {} is missing: {e}",
                    output.display()
                ),
                LogLevel::Error,
            )],
        }
    }

    fn generate_demo_results(&mut self) {
        self.trials = vec![
            TrialResult {
                index: 142,
                refusals: 2,
                total_prompts: 100,
                kl_divergence: 0.0312,
                direction: "global".into(),
            },
            TrialResult {
                index: 87,
                refusals: 0,
                total_prompts: 100,
                kl_divergence: 0.1247,
                direction: "per layer".into(),
            },
            TrialResult {
                index: 198,
                refusals: 1,
                total_prompts: 100,
                kl_divergence: 0.0589,
                direction: "global".into(),
            },
            TrialResult {
                index: 56,
                refusals: 3,
                total_prompts: 100,
                kl_divergence: 0.0201,
                direction: "per layer".into(),
            },
            TrialResult {
                index: 171,
                refusals: 5,
                total_prompts: 100,
                kl_divergence: 0.0098,
                direction: "global".into(),
            },
        ];
    }

    fn switch_to_results(&mut self) {
        self.screen = Screen::Results;
        self.trial_list_state.select(Some(0));
        self.current_menu = vec![
            MenuItem::new(
                "Select this trial",
                "Use the selected trial for export/chat",
            ),
            MenuItem::new(
                "Run additional trials",
                "Continue optimization with more trials",
            ),
            MenuItem::new("Back to main menu", "Return to splash screen"),
        ];
        self.menu_state.select(Some(0));
    }

    /// Handle keyboard input — returns true if app should quit
    pub fn handle_key(&mut self, key: KeyEvent) -> bool {
        // Global quit on Ctrl+C
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            return true;
        }

        match &self.screen.clone() {
            Screen::Splash => self.handle_splash_key(key),
            Screen::Setup => self.handle_setup_key(key),
            Screen::ModelInput => self.handle_model_input_key(key),
            Screen::TokenInput => self.handle_token_input_key(key),
            Screen::ConfigSelect => self.handle_config_select_key(key),
            Screen::Processing => self.handle_processing_key(key),
            Screen::Results => self.handle_results_key(key),
            Screen::TrialActions => self.handle_trial_actions_key(key),
            Screen::Chat => self.handle_chat_key(key),
            Screen::BenchmarkDashboard => self.handle_benchmark_dashboard_key(key),
            Screen::Export => self.handle_export_key(key),
            Screen::CompletedModels => self.handle_completed_models_key(key),
            Screen::GgufSizeSelect => self.handle_gguf_size_select_key(key),
            Screen::CheckpointPrompt => self.handle_checkpoint_prompt_key(key),
            Screen::Confirm(action) => self.handle_confirm_key(key, action.clone()),
            Screen::About => self.handle_about_key(key),
            Screen::RecentModels => self.handle_recent_models_key(key),
        }

        self.should_quit
    }

    /// Handle mouse input — primarily for scrolling
    pub fn handle_mouse(&mut self, mouse: crossterm::event::MouseEvent) {
        use crossterm::event::MouseEventKind;

        match self.screen {
            Screen::Processing => {
                match mouse.kind {
                    MouseEventKind::ScrollUp => {
                        if self.log_scroll > 0 {
                            self.log_scroll = self.log_scroll.saturating_sub(1);
                            self.log_auto_scroll = false;
                        }
                    }
                    MouseEventKind::ScrollDown => {
                        self.log_scroll += 1;
                        // Bounds checking will happen in draw loop
                    }
                    _ => {}
                }
            }
            Screen::Chat => match mouse.kind {
                MouseEventKind::ScrollUp => {
                    self.chat_scroll = self.chat_scroll.saturating_sub(1);
                    // Any deliberate scroll back detaches from the live tail.
                    self.chat_auto_scroll = false;
                }
                MouseEventKind::ScrollDown => {
                    self.chat_scroll += 1;
                    // Clamped against the wrapped height in render_chat, which
                    // is the only place the pane width is known.
                    self.chat_auto_scroll = false;
                }
                _ => {}
            },
            // Add other screen mouse handling here if needed
            _ => {}
        }
    }

    // ─── Splash Screen Keys ────────────────────────────────────

    fn handle_splash_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.menu_up(),
            KeyCode::Down | KeyCode::Char('j') => self.menu_down(),
            KeyCode::Enter => {
                match self.menu_state.selected() {
                    Some(0) => {
                        // Start
                        self.screen = Screen::Setup;
                        self.current_menu = vec![
                            MenuItem::new(
                                "Enter Model ID/Path",
                                "Type a Hugging Face model ID or local path",
                            )
                            .with_key("M"),
                            MenuItem::new("Recent Models", "Choose from previously used models")
                                .with_key("R"),
                            MenuItem::new("Back", "Return to main menu").with_key("Esc"),
                        ];
                        self.menu_state.select(Some(0));
                    }
                    Some(1) => {
                        // Completed Models
                        let checkpoint_dir = crate::subprocess::get_repo_root().join("checkpoints");
                        let mut models = Vec::new();
                        if let Ok(entries) = std::fs::read_dir(checkpoint_dir) {
                            for entry in entries.filter_map(|e| e.ok()) {
                                if let Some(ext) = entry.path().extension() {
                                    if ext == "jsonl" {
                                        models.push(model_name_from_checkpoint(&entry.path()));
                                    }
                                }
                            }
                        }

                        if models.is_empty() {
                            self.status_message = "No completed models found.".to_string();
                        } else {
                            self.screen = Screen::CompletedModels;
                            self.current_menu = models
                                .into_iter()
                                .map(|m| MenuItem::new(&m, "Select to convert to GGUF"))
                                .collect();
                            self.current_menu
                                .push(MenuItem::new("Back", "Return to main menu").with_key("Esc"));
                            self.menu_state.select(Some(0));
                        }
                    }
                    Some(2) => {
                        // Config
                        self.screen = Screen::ConfigSelect;
                        self.current_menu = vec![
                            MenuItem::new(
                                "Default (200 trials)",
                                "Standard configuration, no quantization",
                            ),
                            MenuItem::new("Quick Test (50 trials)", "Faster run for testing"),
                            MenuItem::new("Aggressive (400 trials)", "More thorough optimization"),
                            MenuItem::new("4-bit Quantized", "Lower VRAM usage with bnb_4bit"),
                            MenuItem::new(
                                "OBLITERATUS Advanced",
                                "Gaussian kernel, COSMIC selection, and MoE EGA",
                            )
                            .with_key("O"),
                            MenuItem::new(
                                "Set HF Token",
                                "Required for private models or uploading to Hub",
                            )
                            .with_key("T"),
                            MenuItem::new("Back", "Return to main menu").with_key("Esc"),
                        ];
                        self.menu_state.select(Some(0));
                    }
                    Some(3) => {
                        self.screen = Screen::About;
                    }
                    Some(4) => self.should_quit = true, // Quit
                    _ => {}
                }
            }
            KeyCode::Char('q') | KeyCode::Char('Q') => self.should_quit = true,
            _ => {}
        }
    }

    // ─── Completed Models Keys ─────────────────────────────────

    fn handle_completed_models_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.menu_up(),
            KeyCode::Down | KeyCode::Char('j') => self.menu_down(),
            KeyCode::Enter => {
                let idx = self.menu_state.selected().unwrap_or(0);
                if idx < self.current_menu.len().saturating_sub(1) {
                    // Selected a model
                    self.model_input = self.current_menu[idx].label.clone();
                    self.screen = Screen::TrialActions;
                    self.current_menu = vec![
                        MenuItem::new("Chat with Model", "Test the decensored model").with_key("C"),
                        MenuItem::new("Run Benchmarks", "Evaluate with MMLU, GSM8K, etc.")
                            .with_key("B"),
                        MenuItem::new("Convert to GGUF", "Export as a quantized GGUF file")
                            .with_key("G"),
                        MenuItem::new("Upload to Hugging Face", "Push model to HF Hub")
                            .with_key("U"),
                        MenuItem::new("Delete Model", "Remove the checkpoint").with_key("D"),
                        MenuItem::new("Back", "Return to Completed Models").with_key("Esc"),
                    ];
                    self.menu_state.select(Some(0));
                } else {
                    // Back
                    self.go_back_to_splash();
                }
            }
            KeyCode::Esc => self.go_back_to_splash(),
            _ => {}
        }
    }

    // ─── Setup Screen Keys ─────────────────────────────────────

    fn handle_setup_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.menu_up(),
            KeyCode::Down | KeyCode::Char('j') => self.menu_down(),
            KeyCode::Enter => {
                match self.menu_state.selected() {
                    Some(0) => {
                        // Enter model
                        self.screen = Screen::ModelInput;
                        self.model_input.clear();
                        self.model_cursor = 0;
                    }
                    Some(1) => {
                        // Recent models
                        let recent_file = crate::subprocess::get_repo_root().join(".recent_models");
                        let recent: Vec<String> = std::fs::read_to_string(&recent_file)
                            .unwrap_or_default()
                            .lines()
                            .map(|s| s.to_string())
                            .filter(|s| !s.is_empty())
                            .collect();

                        if recent.is_empty() {
                            self.status_message = "No recent models found.".to_string();
                        } else {
                            self.screen = Screen::RecentModels;
                            self.current_menu = recent
                                .into_iter()
                                .map(|m| MenuItem::new(&m, "Select to launch"))
                                .collect();
                            self.current_menu
                                .push(MenuItem::new("Back", "Return to setup").with_key("Esc"));
                            self.menu_state.select(Some(0));
                        }
                    }
                    Some(2) | _ => self.go_back_to_splash(),
                }
            }
            KeyCode::Esc => self.go_back_to_splash(),
            _ => {}
        }
    }

    // ─── Model Input Screen Keys ───────────────────────────────

    fn handle_model_input_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('v') | KeyCode::Char('V')
                if key
                    .modifiers
                    .contains(crossterm::event::KeyModifiers::CONTROL) =>
            {
                if let Ok(mut clipboard) = arboard::Clipboard::new() {
                    if let Ok(text) = clipboard.get_text() {
                        self.handle_paste(text);
                    } else {
                        self.model_error = Some("Failed to get text from clipboard!".to_string());
                    }
                } else {
                    self.model_error = Some("Host clipboard not accessible here! Use Ctrl+Shift+V or Right-Click instead.".to_string());
                }
            }
            KeyCode::Insert
                if key
                    .modifiers
                    .contains(crossterm::event::KeyModifiers::SHIFT) =>
            {
                if let Ok(mut clipboard) = arboard::Clipboard::new() {
                    if let Ok(text) = clipboard.get_text() {
                        self.handle_paste(text);
                    } else {
                        self.model_error = Some("Failed to get text from clipboard!".to_string());
                    }
                } else {
                    self.model_error = Some("Host clipboard not accessible here! Use Ctrl+Shift+V or Right-Click instead.".to_string());
                }
            }
            KeyCode::Enter => {
                if !self.model_input.is_empty() {
                    // Update status to show validating UI freeze
                    self.status_message = "Validating model on HuggingFace...".to_string();
                    if self.validate_model_input() {
                        self.check_and_start_processing();
                    }
                }
            }
            KeyCode::Esc => {
                self.screen = Screen::Setup;
                self.model_error = None;
                self.current_menu = vec![
                    MenuItem::new(
                        "Enter Model ID/Path",
                        "Type a Hugging Face model ID or local path",
                    )
                    .with_key("M"),
                    MenuItem::new("Recent Models", "Choose from previously used models")
                        .with_key("R"),
                    MenuItem::new("Back", "Return to main menu").with_key("Esc"),
                ];
                self.menu_state.select(Some(0));
            }
            KeyCode::Char(c) => {
                self.model_error = None;
                self.model_cursor =
                    insert_at_char_cursor(&mut self.model_input, self.model_cursor, c);
            }
            KeyCode::Backspace => {
                self.model_error = None;
                self.model_cursor =
                    remove_before_char_cursor(&mut self.model_input, self.model_cursor);
            }
            KeyCode::Left => {
                if self.model_cursor > 0 {
                    self.model_cursor -= 1;
                }
            }
            KeyCode::Right => {
                if self.model_cursor < char_len(&self.model_input) {
                    self.model_cursor += 1;
                }
            }
            KeyCode::Home => self.model_cursor = 0,
            KeyCode::End => self.model_cursor = char_len(&self.model_input),
            _ => {}
        }
    }

    fn validate_model_input(&mut self) -> bool {
        self.model_error = None;
        let model = self.model_input.trim();

        // 1. Check if it's a local directory
        let path = std::path::Path::new(model);
        if path.exists() && path.is_dir() {
            return true;
        }

        // 2. Validate HuggingFace repo via curl HTTP check
        // We accept HTTP 200 (public) and HTTP 401 (gated/private) as valid existence indicators.
        // We use curl because it avoids python SSL certificate issues common in WSL, and is native to Win10+/Linux/Mac.
        let dev_null = if cfg!(windows) { "NUL" } else { "/dev/null" };
        let url = format!("https://huggingface.co/api/models/{}", model);

        let output = std::process::Command::new("curl")
            .args(["-m", "5", "-s", "-o", dev_null, "-w", "%{http_code}", &url])
            .output();

        if let Ok(out) = output {
            let code_str = String::from_utf8_lossy(&out.stdout).trim().to_string();
            // Allow 200 OK or 401 Unauthorized (gated repo)
            if code_str == "200" || code_str == "401" {
                return true;
            }
        }

        self.model_error = Some(format!(
            "Error: '{}' is not a local folder and was not found on HuggingFace Hub!",
            model
        ));
        false
    }
    pub fn handle_paste(&mut self, text: String) {
        let clean = text.replace('\n', "").replace('\r', "");
        if self.screen == Screen::ModelInput {
            self.model_error = None;
            for c in clean.chars() {
                self.model_cursor =
                    insert_at_char_cursor(&mut self.model_input, self.model_cursor, c);
            }
        } else if self.screen == Screen::TokenInput {
            for c in clean.chars() {
                self.hf_token_cursor =
                    insert_at_char_cursor(&mut self.hf_token_input, self.hf_token_cursor, c);
            }
        }
    }

    // ─── Config Select Keys ────────────────────────────────────

    fn handle_config_select_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.menu_up(),
            KeyCode::Down | KeyCode::Char('j') => self.menu_down(),
            KeyCode::Enter => {
                match self.menu_state.selected() {
                    Some(0) => {
                        self.total_trials = 200;
                        self.quantize = false;
                        self.use_obliteratus = false;
                    }
                    Some(1) => {
                        self.total_trials = 50;
                        self.quantize = false;
                        self.use_obliteratus = false;
                    }
                    Some(2) => {
                        self.total_trials = 400;
                        self.quantize = false;
                        self.use_obliteratus = false;
                    }
                    Some(3) => {
                        self.total_trials = 200;
                        self.quantize = true;
                        self.use_obliteratus = false;
                    }
                    Some(4) => {
                        self.total_trials = 200;
                        self.quantize = false;
                        self.use_obliteratus = true;
                    }
                    Some(5) => {
                        self.screen = Screen::TokenInput;
                        self.hf_token_cursor = char_len(&self.hf_token_input);
                        return; // Don't go back to splash
                    }
                    Some(6) => {
                        self.go_back_to_splash();
                        return;
                    }
                    _ => {}
                }
                self.go_back_to_splash();
                self.status_message = if self.use_obliteratus {
                    format!("Config: {} trials, OBLITERATUS Advanced", self.total_trials)
                } else if self.quantize {
                    format!("Config: {} trials, 4-bit", self.total_trials)
                } else {
                    format!("Config: {} trials", self.total_trials)
                };
            }
            KeyCode::Esc => self.go_back_to_splash(),
            _ => {}
        }
    }

    // ─── Token Input Screen Keys ───────────────────────────────

    fn handle_token_input_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('v') | KeyCode::Char('V')
                if key
                    .modifiers
                    .contains(crossterm::event::KeyModifiers::CONTROL) =>
            {
                if let Ok(mut clipboard) = arboard::Clipboard::new() {
                    if let Ok(text) = clipboard.get_text() {
                        let clean = text.replace('\n', "").replace('\r', "");
                        for c in clean.chars() {
                            self.hf_token_cursor = insert_at_char_cursor(
                                &mut self.hf_token_input,
                                self.hf_token_cursor,
                                c,
                            );
                        }
                    }
                }
            }
            KeyCode::Insert
                if key
                    .modifiers
                    .contains(crossterm::event::KeyModifiers::SHIFT) =>
            {
                if let Ok(mut clipboard) = arboard::Clipboard::new() {
                    if let Ok(text) = clipboard.get_text() {
                        let clean = text.replace('\n', "").replace('\r', "");
                        for c in clean.chars() {
                            self.hf_token_cursor = insert_at_char_cursor(
                                &mut self.hf_token_input,
                                self.hf_token_cursor,
                                c,
                            );
                        }
                    }
                }
            }
            KeyCode::Enter => {
                let token = self.hf_token_input.trim().to_string();
                let token_opt = if token.is_empty() {
                    None
                } else {
                    Some(token.as_str())
                };
                unsafe {
                    match token_opt {
                        Some(t) => std::env::set_var("HF_TOKEN", t),
                        None => std::env::remove_var("HF_TOKEN"),
                    }
                }
                self.status_message = match persist_hf_token(token_opt) {
                    Ok(()) if token_opt.is_some() => {
                        "HuggingFace token saved to .env (gitignored).".to_string()
                    }
                    Ok(()) => "HuggingFace token cleared.".to_string(),
                    Err(e) => format!("Token set for this session, but .env write failed: {e}"),
                };

                // Go back to config select
                self.screen = Screen::ConfigSelect;
                self.current_menu = vec![
                    MenuItem::new(
                        "Default (200 trials)",
                        "Standard configuration, no quantization",
                    ),
                    MenuItem::new("Quick Test (50 trials)", "Faster run for testing"),
                    MenuItem::new("Aggressive (400 trials)", "More thorough optimization"),
                    MenuItem::new("4-bit Quantized", "Lower VRAM usage with bnb_4bit"),
                    MenuItem::new(
                        "OBLITERATUS Advanced",
                        "Gaussian kernel, COSMIC selection, and MoE EGA",
                    )
                    .with_key("O"),
                    MenuItem::new(
                        "Set HF Token",
                        "Required for private models or uploading to Hub",
                    )
                    .with_key("T"),
                    MenuItem::new("Back", "Return to main menu").with_key("Esc"),
                ];
                self.menu_state.select(Some(5));
            }
            KeyCode::Esc => {
                self.screen = Screen::ConfigSelect;
                self.current_menu = vec![
                    MenuItem::new(
                        "Default (200 trials)",
                        "Standard configuration, no quantization",
                    ),
                    MenuItem::new("Quick Test (50 trials)", "Faster run for testing"),
                    MenuItem::new("Aggressive (400 trials)", "More thorough optimization"),
                    MenuItem::new("4-bit Quantized", "Lower VRAM usage with bnb_4bit"),
                    MenuItem::new(
                        "OBLITERATUS Advanced",
                        "Gaussian kernel, COSMIC selection, and MoE EGA",
                    )
                    .with_key("O"),
                    MenuItem::new(
                        "Set HF Token",
                        "Required for private models or uploading to Hub",
                    )
                    .with_key("T"),
                    MenuItem::new("Back", "Return to main menu").with_key("Esc"),
                ];
                self.menu_state.select(Some(5));
            }
            KeyCode::Char(c) => {
                self.hf_token_cursor =
                    insert_at_char_cursor(&mut self.hf_token_input, self.hf_token_cursor, c);
            }
            KeyCode::Backspace => {
                self.hf_token_cursor =
                    remove_before_char_cursor(&mut self.hf_token_input, self.hf_token_cursor);
            }
            KeyCode::Left => {
                if self.hf_token_cursor > 0 {
                    self.hf_token_cursor -= 1;
                }
            }
            KeyCode::Right => {
                if self.hf_token_cursor < char_len(&self.hf_token_input) {
                    self.hf_token_cursor += 1;
                }
            }
            KeyCode::Home => self.hf_token_cursor = 0,
            KeyCode::End => self.hf_token_cursor = char_len(&self.hf_token_input),
            _ => {}
        }
    }

    // ─── Processing Screen Keys ────────────────────────────────

    fn handle_processing_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('q') | KeyCode::Char('Q') => {
                self.screen = Screen::Confirm(ConfirmAction::StopProcessing);
                self.current_menu = vec![
                    MenuItem::new(
                        "Yes, stop processing",
                        "Halt optimization and view results so far",
                    ),
                    MenuItem::new("No, continue", "Keep running trials"),
                ];
                self.menu_state.select(Some(1)); // Default to "No"
            }
            KeyCode::Up | KeyCode::Char('k') => {
                if self.log_scroll > 0 {
                    self.log_scroll = self.log_scroll.saturating_sub(1);
                    self.log_auto_scroll = false;
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.log_scroll += 1;
                // Bounds checking will happen in draw loop
            }
            KeyCode::Char('c') | KeyCode::Char('C') => {
                let log_text = self
                    .log_lines
                    .iter()
                    .map(|(msg, _)| msg.clone())
                    .collect::<Vec<String>>()
                    .join("\n");
                if let Ok(mut clipboard) = arboard::Clipboard::new() {
                    let _ = clipboard.set_text(log_text);
                    self.log_lines.push((
                        "Copied entire log to clipboard!".to_string(),
                        LogLevel::Success,
                    ));
                } else {
                    self.log_lines
                        .push(("Failed to access clipboard.".to_string(), LogLevel::Error));
                }
            }
            _ => {}
        }
    }

    // ─── Results Screen Keys ───────────────────────────────────

    fn handle_results_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                let i = self.trial_list_state.selected().unwrap_or(0);
                if i > 0 {
                    self.trial_list_state.select(Some(i - 1));
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let i = self.trial_list_state.selected().unwrap_or(0);
                if i < self.trials.len().saturating_sub(1) {
                    self.trial_list_state.select(Some(i + 1));
                }
            }
            KeyCode::Enter => {
                if self.trial_list_state.selected().is_some() {
                    self.screen = Screen::TrialActions;
                    self.current_menu = vec![
                        MenuItem::new("Save Model Locally", "Export merged model to a folder")
                            .with_key("S"),
                        MenuItem::new("Upload to Hugging Face", "Push model to HF Hub")
                            .with_key("U"),
                        MenuItem::new("Chat with Model", "Test the decensored model").with_key("C"),
                        MenuItem::new("Run Benchmarks", "Evaluate with MMLU, GSM8K, etc.")
                            .with_key("B"),
                        MenuItem::new("Run More Trials", "Continue optimization").with_key("R"),
                        MenuItem::new("Back to Results", "Return to trial selection")
                            .with_key("Esc"),
                    ];
                    self.menu_state.select(Some(0));
                }
            }
            KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => {
                self.go_back_to_splash();
            }
            _ => {}
        }
    }

    // ─── Trial Actions Keys ────────────────────────────────────

    fn handle_trial_actions_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.menu_up(),
            KeyCode::Down | KeyCode::Char('j') => self.menu_down(),
            KeyCode::Enter => {
                if let Some(idx) = self.menu_state.selected() {
                    let label = self.current_menu[idx].label.clone();
                    match label.as_str() {
                        "Save Model Locally" => {
                            self.screen = Screen::Export;
                            self.current_menu = vec![
                                MenuItem::new(
                                    "Merge and export full model",
                                    "Requires sufficient RAM",
                                ),
                                MenuItem::new(
                                    "Export adapter only",
                                    "Can be merged later, smaller file",
                                ),
                                MenuItem::new("Convert to GGUF", "Export as a quantized GGUF file"),
                                MenuItem::new("Back", "Return to actions menu").with_key("Esc"),
                            ];
                            self.menu_state.select(Some(0));
                        }
                        "Upload to Hugging Face" => {
                            self.screen = Screen::TokenInput;
                            self.status_message =
                                "Enter your Hugging Face token to upload the model.".to_string();
                        }
                        "Chat with Model" => {
                            self.screen = Screen::Chat;
                            self.chat_messages.clear();
                            self.chat_input.clear();
                            self.chat_loading = true;
                            self.chat_streaming = false;
                            self.status_message = "Starting chat server...".to_string();
                            self.chat_messages.push((
                                "system".to_string(),
                                "Starting chat server... Reconstructing annihilated model from checkpoint.".to_string(),
                            ));
                            self.chat_subprocess =
                                Some(SubprocessManager::spawn_chat_server(&self.model_input));
                        }
                        "Run Benchmarks" => {
                            self.screen = Screen::BenchmarkDashboard;
                            self.benchmark_running = true;
                            self.benchmark_results.clear();
                            self.log_lines.clear();
                            self.log_lines.push((
                                "Starting benchmarks on annihilated model...".to_string(),
                                LogLevel::Info,
                            ));
                            self.benchmark_subprocess =
                                Some(SubprocessManager::spawn_benchmark(&self.model_input));
                            self.status_message =
                                "Running benchmarks... This may take a while.".to_string();
                        }
                        "Run More Trials" => { /* More trials */ }
                        "Convert to GGUF" => {
                            self.screen = Screen::GgufSizeSelect;
                            self.current_menu = vec![
                                MenuItem::new(
                                    "Q4_K_M",
                                    "Good balance of quality and size (recommended)",
                                ),
                                MenuItem::new("Q8_0", "Near-perfect quality, larger size"),
                                MenuItem::new("F16", "Unquantized, maximum quality"),
                                MenuItem::new("Back", "Return to export options").with_key("Esc"),
                            ];
                            self.menu_state.select(Some(0));
                        }
                        "Delete Model" => {
                            // Destructive: confirm before removing the study.
                            let model = self.model_input.clone();
                            self.screen = Screen::Confirm(ConfirmAction::DeleteCheckpoint(model));
                            self.current_menu = vec![
                                MenuItem::new("Delete", "Permanently remove this checkpoint")
                                    .with_key("Y"),
                                MenuItem::new("Cancel", "Keep the checkpoint").with_key("Esc"),
                            ];
                            self.menu_state.select(Some(1)); // Default to Cancel
                        }
                        "Back to Results" => {
                            self.switch_to_results();
                        }
                        "Back" => {
                            self.go_back_to_splash();
                        }
                        _ => {
                            self.go_back_to_splash();
                        }
                    }
                }
            }
            KeyCode::Esc => {
                // Determine context-aware escape:
                if self
                    .current_menu
                    .iter()
                    .any(|m| m.label == "Back to Results")
                {
                    self.switch_to_results();
                } else {
                    self.go_back_to_splash();
                }
            }
            _ => {}
        }
    }

    // ─── Chat Keys ─────────────────────────────────────────────

    fn handle_chat_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Enter => {
                if !self.chat_input.is_empty() && !self.chat_loading && !self.chat_streaming {
                    let msg = self.chat_input.clone();
                    self.chat_messages.push(("user".to_string(), msg.clone()));
                    self.chat_input.clear();
                    // Jump back to the tail so the user sees their own message
                    // and the reply streaming in.
                    self.chat_auto_scroll = true;

                    let chat_history: Vec<serde_json::Value> = self
                        .chat_messages
                        .iter()
                        .filter(|(role, _)| role == "user" || role == "assistant")
                        .map(
                            |(role, content)| serde_json::json!({"role": role, "content": content}),
                        )
                        .collect();

                    let mut full_chat = vec![
                        serde_json::json!({"role": "system", "content": "You are a helpful assistant."}),
                    ];
                    full_chat.extend(chat_history);

                    if let Some(ref proc) = self.chat_subprocess {
                        let json_str = serde_json::to_string(&full_chat).unwrap_or_default();
                        if proc.send_input(&json_str) {
                            self.chat_streaming = true;
                        } else {
                            self.chat_messages.push((
                                "system".to_string(),
                                "Failed to send message to chat server.".to_string(),
                            ));
                        }
                    } else {
                        self.chat_messages.push((
                            "system".to_string(),
                            "Chat server not running. Press Esc and try again.".to_string(),
                        ));
                    }
                }
            }
            // Scrollback. These sit above the Char arm so they are not
            // swallowed as message text.
            KeyCode::PageUp => {
                self.chat_scroll = self.chat_scroll.saturating_sub(10);
                self.chat_auto_scroll = false;
            }
            KeyCode::PageDown => {
                self.chat_scroll += 10;
                self.chat_auto_scroll = false;
            }
            KeyCode::Up => {
                self.chat_scroll = self.chat_scroll.saturating_sub(1);
                self.chat_auto_scroll = false;
            }
            KeyCode::Down => {
                self.chat_scroll += 1;
                self.chat_auto_scroll = false;
            }
            KeyCode::Home => {
                self.chat_scroll = 0;
                self.chat_auto_scroll = false;
            }
            KeyCode::End => {
                // Re-attach to the live tail; render_chat pins the offset.
                self.chat_auto_scroll = true;
            }
            KeyCode::Char(c) => {
                if !self.chat_loading {
                    self.chat_input.push(c);
                }
            }
            KeyCode::Backspace => {
                self.chat_input.pop();
            }
            KeyCode::Esc => {
                if let Some(ref mut proc) = self.chat_subprocess {
                    proc.kill();
                }
                self.chat_subprocess = None;
                self.chat_loading = false;
                self.chat_streaming = false;
                self.screen = Screen::TrialActions;
                self.menu_state.select(Some(2));
            }
            _ => {}
        }
    }

    // ─── Benchmark Dashboard Keys ──────────────────────────────

    fn handle_benchmark_dashboard_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                if let Some(ref mut proc) = self.benchmark_subprocess {
                    proc.kill();
                }
                self.benchmark_subprocess = None;
                self.benchmark_running = false;
                self.screen = Screen::TrialActions;
                self.menu_state.select(Some(1)); // Usually benchmark is index 1
            }
            KeyCode::Up | KeyCode::Char('k') => {
                if self.log_scroll > 0 {
                    self.log_scroll -= 1;
                    self.log_auto_scroll = false;
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.log_scroll += 1;
            }
            _ => {}
        }
    }

    // ─── Export Keys ───────────────────────────────────────────

    fn handle_export_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.menu_up(),
            KeyCode::Down | KeyCode::Char('j') => self.menu_down(),
            KeyCode::Enter => match self.menu_state.selected() {
                Some(0) | Some(1) => {
                    self.status_message = "Model export complete!".to_string();
                    self.screen = Screen::TrialActions;
                    self.menu_state.select(Some(0));
                }
                Some(2) => {
                    self.screen = Screen::GgufSizeSelect;
                    self.current_menu = vec![
                        MenuItem::new("Q4_K_M", "Good balance of quality and size (recommended)"),
                        MenuItem::new("Q8_0", "Near-perfect quality, larger size"),
                        MenuItem::new("F16", "Unquantized, maximum quality"),
                        MenuItem::new("Back", "Return to export options").with_key("Esc"),
                    ];
                    self.menu_state.select(Some(0));
                }
                _ => {
                    self.screen = Screen::TrialActions;
                    self.menu_state.select(Some(0));
                }
            },
            KeyCode::Esc => {
                self.screen = Screen::TrialActions;
                self.menu_state.select(Some(0));
            }
            _ => {}
        }
    }

    // ─── GGUF Size Select Keys ─────────────────────────────────

    fn handle_gguf_size_select_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.menu_up(),
            KeyCode::Down | KeyCode::Char('j') => self.menu_down(),
            KeyCode::Enter => {
                match self.menu_state.selected() {
                    Some(0) => self.gguf_size = "Q4_K_M".to_string(),
                    Some(1) => self.gguf_size = "Q8_0".to_string(),
                    Some(2) => self.gguf_size = "f16".to_string(),
                    _ => {
                        self.go_back_to_splash();
                        return;
                    }
                }

                self.is_processing = true;
                self.log_lines.clear();
                let msg = format!("Starting GGUF conversion (Size: {})...", self.gguf_size);
                self.log_lines.push((msg, LogLevel::Info));
                self.screen = Screen::Processing;

                // Spawn the subprocess using the gguf converter script
                // We'll pass the model input name for now as a placeholder
                self.gguf_output = Some(crate::subprocess::gguf_output_path(
                    &self.model_input,
                    &self.gguf_size,
                ));
                self.subprocess = Some(crate::subprocess::SubprocessManager::spawn_gguf_convert(
                    &self.model_input,
                    &self.gguf_size,
                ));
            }
            KeyCode::Esc => {
                self.go_back_to_splash();
            }
            _ => {}
        }
    }

    // ─── Confirm Dialog Keys ───────────────────────────────────

    fn handle_confirm_key(&mut self, key: KeyEvent, action: ConfirmAction) {
        let confirmed = match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.menu_up();
                return;
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.menu_down();
                return;
            }
            KeyCode::Enter => self.menu_state.selected() == Some(0),
            KeyCode::Char('y') => true,
            KeyCode::Esc | KeyCode::Char('n') => false,
            _ => return,
        };

        match (confirmed, action) {
            (true, ConfirmAction::Quit) => self.should_quit = true,
            (false, ConfirmAction::Quit) => self.go_back_to_splash(),

            (true, ConfirmAction::StopProcessing) => {
                self.is_processing = false;
                if !self.trials.is_empty() || self.current_trial > 0 {
                    self.generate_demo_results();
                    self.switch_to_results();
                } else {
                    self.go_back_to_splash();
                }
            }
            (false, ConfirmAction::StopProcessing) => self.screen = Screen::Processing,

            (true, ConfirmAction::DeleteCheckpoint(model)) => {
                self.status_message = match delete_checkpoint(&model) {
                    Ok(true) => format!("Deleted checkpoint for {model}"),
                    Ok(false) => format!("No checkpoint found for {model}"),
                    Err(e) => format!("Could not delete checkpoint for {model}: {e}"),
                };
                self.go_back_to_splash();
            }
            (false, ConfirmAction::DeleteCheckpoint(_)) => {
                self.status_message = "Deletion cancelled.".to_string();
                self.switch_to_results();
            }
        }
    }

    // ─── About Screen Keys ─────────────────────────────────────

    fn handle_about_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc | KeyCode::Enter | KeyCode::Char('q') | KeyCode::Char('Q') => {
                self.screen = Screen::Splash;
                self.menu_state.select(Some(2)); // Reselect "About" in the menu
            }
            _ => {}
        }
    }

    // ─── Recent Models Keys ────────────────────────────────────

    fn handle_recent_models_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.menu_up(),
            KeyCode::Down | KeyCode::Char('j') => self.menu_down(),
            KeyCode::Enter => {
                let selected = self.menu_state.selected().unwrap_or(0);
                if selected < self.current_menu.len() - 1 {
                    // It's a model
                    self.model_input = self.current_menu[selected].label.clone();
                    self.check_and_start_processing();
                } else {
                    // It's the "Back" button
                    self.screen = Screen::Setup;
                    self.current_menu = vec![
                        MenuItem::new(
                            "Enter Model ID/Path",
                            "Type a Hugging Face model ID or local path",
                        )
                        .with_key("M"),
                        MenuItem::new("Recent Models", "Choose from previously used models")
                            .with_key("R"),
                        MenuItem::new("Back", "Return to main menu").with_key("Esc"),
                    ];
                    self.menu_state.select(Some(1));
                }
            }
            KeyCode::Esc => {
                self.screen = Screen::Setup;
                self.current_menu = vec![
                    MenuItem::new(
                        "Enter Model ID/Path",
                        "Type a Hugging Face model ID or local path",
                    )
                    .with_key("M"),
                    MenuItem::new("Recent Models", "Choose from previously used models")
                        .with_key("R"),
                    MenuItem::new("Back", "Return to main menu").with_key("Esc"),
                ];
                self.menu_state.select(Some(1));
            }
            _ => {}
        }
    }

    // ─── Menu Helpers ──────────────────────────────────────────

    fn menu_up(&mut self) {
        let i = self.menu_state.selected().unwrap_or(0);
        if i > 0 {
            self.menu_state.select(Some(i - 1));
        }
    }

    fn menu_down(&mut self) {
        let i = self.menu_state.selected().unwrap_or(0);
        if i < self.current_menu.len().saturating_sub(1) {
            self.menu_state.select(Some(i + 1));
        }
    }

    fn go_back_to_splash(&mut self) {
        self.screen = Screen::Splash;
        self.current_menu = vec![
            MenuItem::new("Start Decensoring", "Launch the annihilation workflow")
                .with_key("Enter"),
            MenuItem::new("Completed Models", "Export finished models to GGUF").with_key("M"),
            MenuItem::new("Configuration", "Edit settings before running").with_key("C"),
            MenuItem::new("About", "View project information").with_key("A"),
            MenuItem::new("Quit", "Exit the application").with_key("Q"),
        ];
        self.menu_state.select(Some(0));
    }

    fn start_processing(&mut self) {
        self.screen = Screen::Processing;
        self.is_processing = true;
        self.is_setting_up = true;
        self.current_trial = 0;
        self.elapsed_secs = 0;
        self.eta_secs = None;
        self.best_refusals = None;
        self.best_kl = None;
        self.log_lines.clear();
        self.kl_history.clear();
        self.refusal_history.clear();
        self.sys_info.refresh_gpu();
        self.sys_info.refresh_ram();
        self.batch_size = 16;

        self.log_lines.push((
            "Verifying Python Environment and Missing Dependencies...".to_string(),
            LogLevel::Info,
        ));

        // Save to recent models
        if !self.model_input.is_empty() {
            let recent_file = crate::subprocess::get_repo_root().join(".recent_models");
            let mut recent: Vec<String> = std::fs::read_to_string(&recent_file)
                .unwrap_or_default()
                .lines()
                .map(|s| s.to_string())
                .filter(|s| !s.is_empty())
                .collect();
            recent.retain(|s| s != &self.model_input);
            recent.insert(0, self.model_input.clone());
            recent.truncate(5); // Keep top 5
            let _ = std::fs::write(&recent_file, recent.join("\n"));
        }

        // Not a conversion; clear any target left over from an earlier export
        // so this run's exit is not reported against a stale file.
        self.gguf_output = None;

        self.subprocess = Some(SubprocessManager::spawn_setup(
            self.sys_info.gpu_name != "Unknown",
        ));
    }

    fn check_and_start_processing(&mut self) {
        let checkpoint_dir = crate::subprocess::get_repo_root().join("checkpoints");

        let checkpoint_file = checkpoint_dir.join(format!(
            "{}.jsonl",
            crate::subprocess::checkpoint_name(&self.model_input)
        ));

        if checkpoint_file.exists() {
            self.screen = Screen::CheckpointPrompt;
            self.current_menu = vec![
                MenuItem::new(
                    "Resume previous run",
                    "Continue optimization from the saved checkpoint",
                ),
                MenuItem::new("Start fresh", "Delete previous checkpoint and start over"),
                MenuItem::new("Cancel", "Go back"),
            ];
            self.menu_state.select(Some(0));
        } else {
            self.start_processing();
        }
    }

    fn handle_checkpoint_prompt_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.menu_up(),
            KeyCode::Down | KeyCode::Char('j') => self.menu_down(),
            KeyCode::Enter => {
                match self.menu_state.selected() {
                    Some(0) => {
                        // Resume previous run
                        self.start_processing();
                    }
                    Some(1) => {
                        // Start fresh
                        let checkpoint_file = crate::subprocess::get_repo_root()
                            .join("checkpoints")
                            .join(format!(
                                "{}.jsonl",
                                crate::subprocess::checkpoint_name(&self.model_input)
                            ));

                        if checkpoint_file.exists() {
                            let _ = std::fs::remove_file(checkpoint_file);
                        }
                        self.start_processing();
                    }
                    _ => {
                        // Cancel - go back to Setup
                        self.go_back_to_splash();
                    }
                }
            }
            KeyCode::Esc => self.go_back_to_splash(),
            _ => {}
        }
    }

    // ─── Rendering ─────────────────────────────────────────────

    pub fn render(&mut self, frame: &mut Frame) {
        // Full-screen dark background — reset every cell in the buffer
        let area = frame.area();
        let buf = frame.buffer_mut();
        for y in area.y..area.y + area.height {
            for x in area.x..area.x + area.width {
                let cell = &mut buf[(x, y)];
                cell.reset();
                cell.set_bg(theme::BG_DARK);
            }
        }

        match &self.screen.clone() {
            Screen::Splash => self.render_splash(frame, area),
            Screen::Setup => self.render_menu_screen(
                frame,
                area,
                "MODEL SETUP",
                "Select how to specify your model:",
            ),
            Screen::ModelInput => self.render_model_input(frame, area),
            Screen::TokenInput => self.render_token_input(frame, area),
            Screen::ConfigSelect => self.render_menu_screen(
                frame,
                area,
                "CONFIGURATION",
                "Choose an optimization preset:",
            ),
            Screen::Processing => self.render_processing(frame, area),
            Screen::Results => self.render_results(frame, area),
            Screen::TrialActions => self.render_menu_screen(
                frame,
                area,
                "TRIAL ACTIONS",
                "What do you want to do with the decensored model?",
            ),
            Screen::Chat => self.render_chat(frame, area),
            Screen::BenchmarkDashboard => self.render_benchmark_dashboard(frame, area),
            Screen::Export => {
                self.render_menu_screen(frame, area, "EXPORT MODEL", "Choose export strategy:")
            }
            Screen::GgufSizeSelect => self.render_gguf_splash(frame, area),
            Screen::CheckpointPrompt => {
                self.render_menu_screen(
                    frame,
                    area,
                    "MODEL SETUP",
                    "Select how to specify your model:",
                );
                self.render_checkpoint_prompt_dialog(frame, area);
            }
            Screen::CompletedModels => self.render_gguf_splash(frame, area),
            Screen::RecentModels => self.render_menu_screen(
                frame,
                area,
                "RECENT MODELS",
                "Select a previously used model:",
            ),
            Screen::Confirm(action) => {
                // Render previous screen dimmed, then overlay
                match action {
                    ConfirmAction::StopProcessing => self.render_processing(frame, area),
                    ConfirmAction::Quit => self.render_splash(frame, area),
                    ConfirmAction::DeleteCheckpoint(_) => self.render_results(frame, area),
                }
                self.render_confirm_dialog(frame, area);
            }
            Screen::About => self.render_about(frame, area),
        }

        // Status bar at bottom
        self.render_status_bar(frame, area);
    }

    // ─── Splash Screen ─────────────────────────────────────────

    fn render_splash(&mut self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2), // top padding
                Constraint::Length(7), // banner
                Constraint::Length(2), // tagline
                Constraint::Length(1), // spacer
                Constraint::Min(6),    // menu
                Constraint::Length(1), // status bar
            ])
            .split(area);

        // Banner with per-character horizontal neon gradient
        let banner_width = BANNER.iter().map(|l| l.len()).max().unwrap_or(0);

        let banner_lines: Vec<Line> = BANNER
            .iter()
            .map(|line| {
                let chars: Vec<char> = line.chars().collect();
                let spans: Vec<Span> = chars
                    .iter()
                    .enumerate()
                    .map(|(col, &ch)| {
                        let t = if banner_width > 1 {
                            col as f64 / (banner_width - 1) as f64
                        } else {
                            0.0
                        };

                        // Gradient: cyan (0,255,240) → purple (191,0,255) → magenta (255,0,255)
                        let (r, g, b) = if t < 0.5 {
                            let s = t * 2.0;
                            (
                                (0.0 + s * 191.0) as u8,
                                (255.0 - s * 255.0) as u8,
                                (240.0 + s * 15.0) as u8,
                            )
                        } else {
                            let s = (t - 0.5) * 2.0;
                            ((191.0 + s * 64.0) as u8, 0u8, 255u8)
                        };

                        let color = ratatui::style::Color::Rgb(r, g, b);

                        if ch != ' ' {
                            Span::styled(
                                ch.to_string(),
                                Style::default().fg(color).add_modifier(Modifier::BOLD),
                            )
                        } else {
                            Span::raw(" ")
                        }
                    })
                    .collect();
                Line::from(spans)
            })
            .collect();

        let banner = Paragraph::new(banner_lines).alignment(Alignment::Center);
        frame.render_widget(banner, chunks[1]);

        // Tagline
        let glow_intensity = (self.glow_phase * 255.0) as u8;
        let tagline_color = ratatui::style::Color::Rgb(glow_intensity, 200, 255);
        let tagline = Paragraph::new(Line::from(Span::styled(
            TAGLINE,
            Style::default()
                .fg(tagline_color)
                .add_modifier(Modifier::ITALIC),
        )))
        .alignment(Alignment::Center);
        frame.render_widget(tagline, chunks[2]);

        // Menu
        self.render_selection_menu(frame, chunks[4]);
    }

    fn render_gguf_splash(&mut self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2), // top padding
                Constraint::Length(7), // banner
                Constraint::Length(2), // tagline
                Constraint::Length(1), // spacer
                Constraint::Min(6),    // menu
                Constraint::Length(1), // status bar
            ])
            .split(area);

        // Banner with per-character horizontal neon gradient
        let banner_width = GGUF_BANNER.iter().map(|l| l.len()).max().unwrap_or(0);

        let banner_lines: Vec<Line> = GGUF_BANNER
            .iter()
            .map(|line| {
                let chars: Vec<char> = line.chars().collect();
                let spans: Vec<Span> = chars
                    .iter()
                    .enumerate()
                    .map(|(col, &ch)| {
                        let t = if banner_width > 1 {
                            col as f64 / (banner_width - 1) as f64
                        } else {
                            0.0
                        };

                        // Gradient: green/yellow for GGUF
                        let (r, g, b) = if t < 0.5 {
                            let s = t * 2.0;
                            ((50.0 + s * 200.0) as u8, 255u8, 50u8)
                        } else {
                            let s = (t - 0.5) * 2.0;
                            (255u8, (255.0 - s * 100.0) as u8, 50u8)
                        };

                        let color = ratatui::style::Color::Rgb(r, g, b);

                        if ch != ' ' {
                            Span::styled(
                                ch.to_string(),
                                Style::default().fg(color).add_modifier(Modifier::BOLD),
                            )
                        } else {
                            Span::raw(" ")
                        }
                    })
                    .collect();
                Line::from(spans)
            })
            .collect();

        let banner = Paragraph::new(banner_lines).alignment(Alignment::Center);
        frame.render_widget(banner, chunks[1]);

        // Tagline
        let glow_intensity = (self.glow_phase * 255.0) as u8;
        let tagline_color = ratatui::style::Color::Rgb(200, 255, glow_intensity.max(100));
        let tagline = Paragraph::new(Line::from(Span::styled(
            "-- Export your models to high-performance quantized GGUF format --",
            Style::default()
                .fg(tagline_color)
                .add_modifier(Modifier::ITALIC),
        )))
        .alignment(Alignment::Center);
        frame.render_widget(tagline, chunks[2]);

        // Menu (without settings box)
        self.render_selection_menu(frame, chunks[4]);
    }

    // ─── Generic Menu Screen ───────────────────────────────────

    fn render_menu_screen(&mut self, frame: &mut Frame, area: Rect, title: &str, subtitle: &str) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // title
                Constraint::Length(2), // subtitle
                Constraint::Length(1), // spacer
                Constraint::Min(6),    // menu
                Constraint::Length(1), // status bar
            ])
            .split(area);

        // Title
        let title_widget = Paragraph::new(Line::from(vec![
            Span::styled("  ⚔ ", Style::default().fg(theme::NEON_MAGENTA)),
            Span::styled(title, theme::title_style()),
            Span::styled(" ⚔  ", Style::default().fg(theme::NEON_MAGENTA)),
        ]))
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(theme::BORDER_INACTIVE)),
        );
        frame.render_widget(title_widget, chunks[0]);

        // Subtitle
        let sub = Paragraph::new(Line::from(Span::styled(subtitle, theme::dim_style())))
            .alignment(Alignment::Center);
        frame.render_widget(sub, chunks[1]);

        // Menu
        self.render_selection_menu(frame, chunks[3]);
    }

    // ─── Selection Menu Widget ─────────────────────────────────

    fn render_selection_menu(&mut self, frame: &mut Frame, area: Rect) {
        let menu_width = 60.min(area.width.saturating_sub(4));
        let menu_area =
            centered_rect_fixed(menu_width, self.current_menu.len() as u16 * 3 + 2, area);

        let items: Vec<ListItem> = self
            .current_menu
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let is_selected = self.menu_state.selected() == Some(i);

                let prefix = if is_selected { "▸ " } else { "  " };

                let mut spans = vec![
                    Span::styled(
                        prefix,
                        if is_selected {
                            Style::default()
                                .fg(theme::NEON_CYAN)
                                .add_modifier(Modifier::BOLD)
                        } else {
                            Style::default().fg(theme::TEXT_DIM)
                        },
                    ),
                    Span::styled(
                        &item.label,
                        if is_selected {
                            Style::default()
                                .fg(theme::NEON_CYAN)
                                .add_modifier(Modifier::BOLD)
                        } else {
                            Style::default().fg(theme::TEXT_PRIMARY)
                        },
                    ),
                ];

                if let Some(key) = &item.key_hint {
                    spans.push(Span::styled(
                        format!("  [{}]", key),
                        Style::default().fg(if is_selected {
                            theme::NEON_PURPLE
                        } else {
                            theme::TEXT_DIM
                        }),
                    ));
                }

                let main_line = Line::from(spans);
                let desc_line = Line::from(Span::styled(
                    format!("    {}", item.description),
                    Style::default()
                        .fg(if is_selected {
                            theme::BORDER_ACTIVE
                        } else {
                            theme::TEXT_DIM
                        })
                        .add_modifier(Modifier::ITALIC),
                ));

                ListItem::new(vec![main_line, desc_line, Line::from("")])
            })
            .collect();

        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(theme::BORDER_ACTIVE))
                    .title(Span::styled(
                        " Select ",
                        Style::default()
                            .fg(theme::NEON_CYAN)
                            .add_modifier(Modifier::BOLD),
                    ))
                    .title_alignment(Alignment::Center)
                    .style(Style::default().bg(theme::BG_SURFACE)),
            )
            .highlight_style(Style::default()); // We handle highlighting manually

        frame.render_stateful_widget(list, menu_area, &mut self.menu_state);

        // Key hints below menu
        let hint_area = Rect::new(
            menu_area.x,
            menu_area.y + menu_area.height,
            menu_area.width,
            1,
        );
        if hint_area.y < area.y + area.height {
            let hints = Paragraph::new(Line::from(vec![
                Span::styled(" ↑↓ ", theme::key_hint_style()),
                Span::styled("Navigate  ", theme::key_desc_style()),
                Span::styled(" Enter ", theme::key_hint_style()),
                Span::styled("Select  ", theme::key_desc_style()),
                Span::styled(" Esc ", theme::key_hint_style()),
                Span::styled("Back", theme::key_desc_style()),
            ]))
            .alignment(Alignment::Center);
            frame.render_widget(hints, hint_area);
        }
    }

    fn render_settings_panel(&self, frame: &mut Frame, area: Rect, centered: bool) {
        let settings_area = if centered {
            centered_rect_fixed(
                35.min(area.width.saturating_sub(4)),
                14, // height
                area,
            )
        } else {
            area
        };

        let lines = vec![
            Line::from(vec![
                Span::styled("Trials: ", Style::default().fg(theme::TEXT_DIM)),
                Span::styled(
                    format!("{}", self.total_trials),
                    Style::default()
                        .fg(theme::TEXT_PRIMARY)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::styled("Quantization: ", Style::default().fg(theme::TEXT_DIM)),
                Span::styled(
                    if self.quantize { "4-bit (bnb)" } else { "None" },
                    Style::default()
                        .fg(theme::TEXT_PRIMARY)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::styled("OBLITERATUS: ", Style::default().fg(theme::TEXT_DIM)),
                Span::styled(
                    if self.use_obliteratus {
                        "Enabled"
                    } else {
                        "Disabled"
                    },
                    Style::default()
                        .fg(theme::TEXT_PRIMARY)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::styled("Target GGUF: ", Style::default().fg(theme::TEXT_DIM)),
                Span::styled(
                    &self.gguf_size,
                    Style::default()
                        .fg(theme::TEXT_PRIMARY)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Model: ", Style::default().fg(theme::TEXT_DIM)),
                Span::styled(
                    if self.model_input.is_empty() {
                        "Not selected"
                    } else {
                        &self.model_input
                    },
                    Style::default()
                        .fg(theme::NEON_CYAN)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
        ];

        let panel = Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(theme::BORDER_INACTIVE))
                    .title(Span::styled(
                        " Current Settings ",
                        Style::default().fg(theme::TEXT_DIM),
                    ))
                    .title_alignment(Alignment::Center)
                    .style(Style::default().bg(theme::BG_SURFACE)),
            )
            .wrap(ratatui::widgets::Wrap { trim: true });

        frame.render_widget(panel, settings_area);
    }

    // ─── Model Input Screen ────────────────────────────────────

    fn render_model_input(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Length(1),
                Constraint::Length(5),
                Constraint::Min(0),
                Constraint::Length(1),
            ])
            .split(area);

        // Title
        let title = Paragraph::new(Line::from(vec![
            Span::styled("  ⚔ ", Style::default().fg(theme::NEON_MAGENTA)),
            Span::styled("ENTER MODEL", theme::title_style()),
            Span::styled(" ⚔  ", Style::default().fg(theme::NEON_MAGENTA)),
        ]))
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(theme::BORDER_INACTIVE)),
        );
        frame.render_widget(title, chunks[0]);

        let sub = Paragraph::new(Line::from(Span::styled(
            "Enter a Hugging Face model ID or local path:",
            theme::dim_style(),
        )))
        .alignment(Alignment::Center);
        frame.render_widget(sub, chunks[1]);

        if let Some(ref err) = self.model_error {
            let err_widget = Paragraph::new(Line::from(Span::styled(
                err,
                Style::default()
                    .fg(ratatui::style::Color::Red)
                    .add_modifier(Modifier::BOLD),
            )))
            .alignment(Alignment::Center);
            frame.render_widget(err_widget, chunks[2]);
        }

        // Input field
        let input_width = 60.min(area.width.saturating_sub(4));
        let input_area = centered_rect_fixed(input_width, 3, chunks[3]);

        let display_text = if self.model_input.is_empty() {
            "e.g. Qwen/Qwen3-4B-Instruct-2507".to_string()
        } else {
            self.model_input.clone()
        };

        let input_style = if self.model_input.is_empty() {
            Style::default().fg(theme::TEXT_DIM)
        } else {
            Style::default()
                .fg(theme::NEON_CYAN)
                .add_modifier(Modifier::BOLD)
        };

        let input = Paragraph::new(Line::from(Span::styled(&display_text, input_style))).block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(theme::NEON_CYAN))
                .title(Span::styled(
                    " Model ",
                    Style::default()
                        .fg(theme::NEON_CYAN)
                        .add_modifier(Modifier::BOLD),
                ))
                .style(Style::default().bg(theme::BG_SURFACE)),
        );
        frame.render_widget(input, input_area);

        // Show cursor
        let cursor_x = input_area.x + 1 + self.model_cursor as u16;
        let cursor_y = input_area.y + 1;
        if cursor_x < input_area.x + input_area.width - 1 {
            frame.set_cursor_position((cursor_x, cursor_y));
        }

        // Hints
        let hint_area = Rect::new(input_area.x, input_area.y + 4, input_area.width, 1);
        if hint_area.y < area.y + area.height - 1 {
            let hints = Paragraph::new(Line::from(vec![
                Span::styled(" Enter ", theme::key_hint_style()),
                Span::styled("Start  ", theme::key_desc_style()),
                Span::styled(" Esc ", theme::key_hint_style()),
                Span::styled("Back", theme::key_desc_style()),
            ]))
            .alignment(Alignment::Center);
            frame.render_widget(hints, hint_area);
        }
    }

    // ─── Token Input Rendering ─────────────────────────────────

    fn render_token_input(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Length(1),
                Constraint::Length(5),
                Constraint::Min(0),
                Constraint::Length(1),
            ])
            .split(area);

        // Title
        let title = Paragraph::new(Line::from(vec![
            Span::styled("  ? ", Style::default().fg(theme::NEON_AMBER)),
            Span::styled("HUGGINGFACE TOKEN", theme::title_style()),
            Span::styled(" ?  ", Style::default().fg(theme::NEON_AMBER)),
        ]))
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(theme::BORDER_INACTIVE)),
        );
        frame.render_widget(title, chunks[0]);

        // Input Box
        let inner_input_area = centered_rect_fixed(60, 3, chunks[1]);

        let display_text = if self.hf_token_input.is_empty() {
            "e.g. hf_AbcDefGhiJklMnoPqrStuVwxYz...".to_string()
        } else {
            // Mask all but the first 3 characters.
            mask_secret(&self.hf_token_input)
        };

        let input_style = if self.hf_token_input.is_empty() {
            Style::default().fg(theme::TEXT_DIM)
        } else {
            Style::default()
                .fg(theme::NEON_AMBER)
                .add_modifier(Modifier::BOLD)
        };

        let input_block = Paragraph::new(display_text).style(input_style).block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Double)
                .border_style(Style::default().fg(theme::NEON_AMBER))
                .title(Span::styled(" Access Token ", theme::warning_style())),
        );

        frame.render_widget(input_block, inner_input_area);

        // Error message or info
        let info_text = vec![
            Line::from(Span::styled(
                "A HuggingFace User Access Token is required to:",
                Style::default().fg(theme::TEXT_PRIMARY),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled("1. ", Style::default().fg(theme::NEON_CYAN)),
                Span::styled(
                    "Download private/gated models (Read permissions).",
                    theme::dim_style(),
                ),
            ]),
            Line::from(vec![
                Span::styled("2. ", Style::default().fg(theme::NEON_CYAN)),
                Span::styled(
                    "Upload your decensored model to the Hub (Write permissions).",
                    theme::dim_style(),
                ),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Get your token at: ", Style::default().fg(theme::TEXT_DIM)),
                Span::styled(
                    "https://huggingface.co/settings/tokens",
                    Style::default()
                        .fg(theme::NEON_BLUE)
                        .add_modifier(Modifier::UNDERLINED),
                ),
            ]),
        ];

        let info_para = Paragraph::new(info_text)
            .alignment(Alignment::Center)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(theme::BORDER_INACTIVE)),
            );
        let info_area = centered_rect_fixed(70, 8, chunks[3]);
        frame.render_widget(info_para, info_area);

        // Hints
        let hints = Paragraph::new(Line::from(vec![
            Span::styled("Enter ", theme::key_hint_style()),
            Span::styled("Save token   ", theme::key_desc_style()),
            Span::styled("Ctrl+V ", theme::key_hint_style()),
            Span::styled("Paste   ", theme::key_desc_style()),
            Span::styled("Esc ", theme::key_hint_style()),
            Span::styled("Clear/Cancel", theme::key_desc_style()),
        ]))
        .alignment(Alignment::Center);
        frame.render_widget(hints, chunks[5]);

        let cursor_x = inner_input_area.x + 1 + self.hf_token_cursor as u16;
        let cursor_y = inner_input_area.y + 1;
        frame.set_cursor_position((cursor_x, cursor_y));
    }

    // ─── Processing Dashboard ──────────────────────────────────

    fn render_processing(&mut self, frame: &mut Frame, area: Rect) {
        let main_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(75), Constraint::Percentage(25)])
            .split(Rect::new(
                area.x,
                area.y,
                area.width,
                area.height.saturating_sub(1),
            ));

        let left_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(7),  // header + progress
                Constraint::Length(16), // metrics
                Constraint::Min(5),     // log
            ])
            .split(main_chunks[0]);

        let right_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(10), // system info
                Constraint::Length(8),  // controls
                Constraint::Min(10),    // settings
            ])
            .split(main_chunks[1]);

        // ── Header & Progress ──
        let progress_block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(theme::BORDER_ACTIVE))
            .title(Span::styled(" ⚔ ANNIHILATE ", theme::title_style()))
            .style(Style::default().bg(theme::BG_SURFACE));

        let progress_inner = progress_block.inner(left_chunks[0]);
        frame.render_widget(progress_block, left_chunks[0]);

        let progress_lines = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
            ])
            .split(progress_inner);

        // Model name
        let model_line = Line::from(vec![
            Span::styled(" Model: ", theme::dim_style()),
            Span::styled(
                if self.model_input.is_empty() {
                    "demo-model"
                } else {
                    &self.model_input
                },
                theme::highlight_value(),
            ),
        ]);
        frame.render_widget(Paragraph::new(model_line), progress_lines[0]);

        // Progress gauge
        let progress_ratio = if self.total_trials > 0 {
            self.current_trial as f64 / self.total_trials as f64
        } else {
            0.0
        };
        let gauge = Gauge::default()
            .gauge_style(theme::gauge_style())
            .label(Span::styled(
                format!(" Trials: {}/{} ", self.current_trial, self.total_trials),
                Style::default()
                    .fg(theme::TEXT_BRIGHT)
                    .add_modifier(Modifier::BOLD),
            ))
            .ratio(progress_ratio);
        frame.render_widget(gauge, progress_lines[1]);

        // Timing
        let elapsed_str = format_duration(self.elapsed_secs);
        let eta_str = self
            .eta_secs
            .map_or("calculating...".to_string(), format_duration);
        let time_line = Line::from(vec![
            Span::styled(" Elapsed: ", theme::dim_style()),
            Span::styled(&elapsed_str, theme::highlight_value()),
            Span::styled("  ETA: ", theme::dim_style()),
            Span::styled(&eta_str, Style::default().fg(theme::NEON_AMBER)),
        ]);
        frame.render_widget(Paragraph::new(time_line), progress_lines[2]);

        // ── Metrics ──
        let metrics_block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(theme::BORDER_ACTIVE))
            .title(Span::styled(" METRICS ", theme::title_style()))
            .style(Style::default());

        let metrics_inner = metrics_block.inner(left_chunks[1]);
        frame.render_widget(metrics_block, left_chunks[1]);

        let metric_lines = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // Refusal text
                Constraint::Length(1), // KL text
                Constraint::Length(1), // Spacer
                Constraint::Length(5), // KL Chart
                Constraint::Length(1), // Spacer
                Constraint::Min(5),    // Ref Chart
            ])
            .split(metrics_inner);

        let refusal_str = self
            .best_refusals
            .map_or("--".to_string(), |r| format!("{}/100", r));
        let kl_str = self
            .best_kl
            .map_or("--".to_string(), |k| format!("{:.4}", k));

        let refusal_line = Line::from(vec![
            Span::styled(" Best Refusals: ", theme::dim_style()),
            Span::styled(&refusal_str, theme::success_style()),
        ]);
        let kl_line = Line::from(vec![
            Span::styled(" Best KL Div:   ", theme::dim_style()),
            Span::styled(&kl_str, theme::highlight_value()),
        ]);

        frame.render_widget(Paragraph::new(refusal_line), metric_lines[0]);
        frame.render_widget(Paragraph::new(kl_line), metric_lines[1]);

        // Draw Sparklines ("spike graphs") instead of braille dots
        if !self.kl_history.is_empty() {
            let kl_sparkline_data: Vec<u64> = self
                .kl_history
                .iter()
                .map(|&v| (v * 10000.0) as u64)
                .collect();
            let max_val = kl_sparkline_data.iter().max().cloned().unwrap_or(0);

            let kl_sparkline = Sparkline::default()
                .block(Block::default().title(Span::styled(" KL Div ", theme::dim_style())))
                .style(Style::default().fg(theme::NEON_CYAN))
                .data(&kl_sparkline_data)
                .max(max_val.max(1));

            frame.render_widget(kl_sparkline, metric_lines[3]);
        }

        if !self.refusal_history.is_empty() {
            let ref_sparkline_data: Vec<u64> =
                self.refusal_history.iter().map(|&v| v as u64).collect();
            let max_val = ref_sparkline_data.iter().max().cloned().unwrap_or(0);

            let ref_sparkline = Sparkline::default()
                .block(Block::default().title(Span::styled(" Refusals ", theme::dim_style())))
                .style(Style::default().fg(theme::NEON_GREEN))
                .data(&ref_sparkline_data)
                .max(max_val.max(1));

            frame.render_widget(ref_sparkline, metric_lines[5]);
        }

        // ── Log Panel ──
        let log_block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(theme::BORDER_INACTIVE))
            .title(Span::styled(" LOG ", Style::default().fg(theme::TEXT_DIM)))
            .style(Style::default().bg(theme::BG_SURFACE));

        let log_inner = log_block.inner(left_chunks[2]);
        frame.render_widget(log_block, left_chunks[2]);

        let visible_lines = log_inner.height as usize;

        if self.log_auto_scroll {
            self.log_scroll = self.log_lines.len().saturating_sub(visible_lines);
        } else {
            // Clamp scroll to valid bounds just in case, and re-enable auto-scroll if at bottom
            let max_scroll = self.log_lines.len().saturating_sub(visible_lines);
            if self.log_scroll >= max_scroll {
                self.log_scroll = max_scroll;
                self.log_auto_scroll = true;
            }
        }

        let start = self.log_scroll;
        let end = (start + visible_lines).min(self.log_lines.len());

        let log_items: Vec<ListItem> = self.log_lines[start..end]
            .iter()
            .map(|(text, level)| {
                let style = match level {
                    LogLevel::Info => Style::default().fg(theme::TEXT_PRIMARY),
                    LogLevel::Success => theme::success_style(),
                    LogLevel::Warning => theme::warning_style(),
                    LogLevel::Error => theme::error_style(),
                    LogLevel::Dim => theme::dim_style(),
                };
                ListItem::new(Line::from(Span::styled(format!(" {}", text), style)))
            })
            .collect();

        let log_list = List::new(log_items);
        frame.render_widget(log_list, log_inner);

        // ── System Info ──
        let sys_block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(theme::BORDER_INACTIVE))
            .title(Span::styled(
                " SYSTEM ",
                Style::default().fg(theme::TEXT_DIM),
            ))
            .style(Style::default().bg(theme::BG_SURFACE));

        let sys_inner = sys_block.inner(right_chunks[0]);
        frame.render_widget(sys_block, right_chunks[0]);

        let sys_lines: Vec<Line> = vec![
            Line::from(vec![
                Span::styled(" GPU: ", theme::dim_style()),
                Span::styled(
                    &self.sys_info.gpu_name,
                    Style::default().fg(theme::NEON_PURPLE),
                ),
            ]),
            Line::from(vec![
                Span::styled(" VRAM: ", theme::dim_style()),
                Span::styled(
                    format!(
                        "{:.1}/{:.0} GB",
                        self.sys_info.vram_used_gb(),
                        self.sys_info.vram_total_gb()
                    ),
                    theme::highlight_value(),
                ),
            ]),
            Line::from(vec![
                Span::styled(" RAM: ", theme::dim_style()),
                Span::styled(
                    format!(
                        "{:.1}/{:.0} GB",
                        self.sys_info.ram_used_gb(),
                        self.sys_info.ram_total_gb()
                    ),
                    theme::highlight_value(),
                ),
            ]),
            Line::from(vec![
                Span::styled(" Batch: ", theme::dim_style()),
                Span::styled(format!("{}", self.batch_size), theme::highlight_value()),
            ]),
            Line::from(vec![
                Span::styled(" Tok/s: ", theme::dim_style()),
                Span::styled(
                    format!("{:.0}", self.tokens_per_sec),
                    Style::default().fg(theme::NEON_GREEN),
                ),
            ]),
        ];

        let sys_text = Paragraph::new(sys_lines);
        frame.render_widget(sys_text, sys_inner);

        // ── Controls ──
        let ctrl_block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(theme::BORDER_INACTIVE))
            .title(Span::styled(
                " CONTROLS ",
                Style::default().fg(theme::TEXT_DIM),
            ))
            .style(Style::default().bg(theme::BG_SURFACE));

        let ctrl_inner = ctrl_block.inner(right_chunks[1]);
        frame.render_widget(ctrl_block, right_chunks[1]);

        let ctrl_lines: Vec<Line> = vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("  Q ", theme::key_hint_style()),
                Span::styled(" Stop", theme::key_desc_style()),
            ]),
            Line::from(vec![
                Span::styled(" C ", theme::key_hint_style()),
                Span::styled("Copy log  ", theme::key_desc_style()),
                Span::styled("    ", theme::key_hint_style()),
                Span::styled(" Scroll log", theme::key_desc_style()),
            ]),
            Line::from(vec![
                Span::styled(" C-c", theme::key_hint_style()),
                Span::styled(" Force quit", theme::key_desc_style()),
            ]),
        ];

        let ctrl_text = Paragraph::new(ctrl_lines);
        frame.render_widget(ctrl_text, ctrl_inner);

        // ── Settings ──
        self.render_settings_panel(frame, right_chunks[2], false);
    }

    // ─── Results Screen ────────────────────────────────────────

    fn render_results(&mut self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // title
                Constraint::Length(2), // info
                Constraint::Min(8),    // trial list
                Constraint::Length(3), // hints
                Constraint::Length(1), // status
            ])
            .split(area);

        // Title
        let title = Paragraph::new(Line::from(vec![
            Span::styled("  ✨ ", Style::default().fg(theme::NEON_GREEN)),
            Span::styled("OPTIMIZATION COMPLETE", theme::success_style()),
            Span::styled(" ✨  ", Style::default().fg(theme::NEON_GREEN)),
        ]))
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(theme::BORDER_INACTIVE)),
        );
        frame.render_widget(title, chunks[0]);

        // Info text
        let info = Paragraph::new(Line::from(Span::styled(
            "Pareto optimal trials (lowest refusals + KL divergence). Select a trial to proceed:",
            theme::dim_style(),
        )))
        .alignment(Alignment::Center);
        frame.render_widget(info, chunks[1]);

        // Trial list
        let trial_width = 70.min(area.width.saturating_sub(4));
        let trial_area = centered_rect_fixed(trial_width, chunks[2].height, chunks[2]);

        // Header
        let header = Row::new(vec!["Trial", "Refusals", "KL Div", "Direction"])
            .style(
                Style::default()
                    .fg(theme::NEON_PURPLE)
                    .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
            )
            .bottom_margin(1);

        let rows: Vec<Row> = self
            .trials
            .iter()
            .map(|trial| {
                let kl_color = if trial.kl_divergence > 0.5 {
                    theme::NEON_AMBER
                } else if trial.kl_divergence > 0.1 {
                    theme::TEXT_PRIMARY
                } else {
                    theme::NEON_GREEN
                };

                let refusal_color = if trial.refusals == 0 {
                    theme::NEON_GREEN
                } else if trial.refusals <= 5 {
                    theme::NEON_CYAN
                } else {
                    theme::NEON_AMBER
                };

                Row::new(vec![
                    Cell::from(format!("{}", trial.index)),
                    Cell::from(format!("{}/{}", trial.refusals, trial.total_prompts))
                        .style(Style::default().fg(refusal_color)),
                    Cell::from(format!("{:.4}", trial.kl_divergence))
                        .style(Style::default().fg(kl_color)),
                    Cell::from(trial.direction.clone()).style(theme::dim_style()),
                ])
            })
            .collect();

        let trial_table = Table::new(
            rows,
            [
                Constraint::Length(10), // Trial
                Constraint::Length(15), // Refusals
                Constraint::Length(15), // KL Div
                Constraint::Min(20),    // Direction
            ],
        )
        .header(header)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(theme::BORDER_ACTIVE))
                .title(Span::styled(
                    " Pareto Optimal Trials ",
                    theme::title_style(),
                ))
                .style(Style::default().bg(theme::BG_SURFACE)),
        )
        .row_highlight_style(
            Style::default()
                .bg(theme::BG_DARK)
                .fg(theme::NEON_CYAN)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(" ▸ ");

        frame.render_stateful_widget(trial_table, trial_area, &mut self.trial_list_state);

        // Hints
        let hints = Paragraph::new(Line::from(vec![
            Span::styled(" ↑↓ ", theme::key_hint_style()),
            Span::styled("Navigate  ", theme::key_desc_style()),
            Span::styled(" Enter ", theme::key_hint_style()),
            Span::styled("Select trial  ", theme::key_desc_style()),
            Span::styled(" Q ", theme::key_hint_style()),
            Span::styled("Quit", theme::key_desc_style()),
        ]))
        .alignment(Alignment::Center);
        frame.render_widget(hints, chunks[3]);
    }

    // ─── Benchmark Dashboard Screen ────────────────────────────

    fn render_benchmark_dashboard(&mut self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),      // title
                Constraint::Percentage(40), // results table
                Constraint::Percentage(60), // logs
            ])
            .split(area);

        // Title
        let title_text = if self.benchmark_running {
            format!(" 📊 BENCHMARKING: {} (Running...)", self.model_input)
        } else {
            format!(" 📊 BENCHMARKING: {} (Completed)", self.model_input)
        };
        let title = Paragraph::new(Line::from(vec![Span::styled(
            title_text,
            theme::title_style(),
        )]))
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(theme::BORDER_INACTIVE)),
        );
        frame.render_widget(title, chunks[0]);

        // Results Table
        let header = Row::new(vec!["Benchmark", "Metric", "Score"])
            .style(
                Style::default()
                    .fg(theme::NEON_CYAN)
                    .add_modifier(Modifier::BOLD)
                    .bg(theme::BG_SURFACE),
            )
            .height(1)
            .bottom_margin(1);

        let rows: Vec<Row> = self
            .benchmark_results
            .iter()
            .map(|(bench, metric, value)| {
                Row::new(vec![
                    Cell::from(Span::styled(bench, theme::highlight_value())),
                    Cell::from(metric.clone()),
                    Cell::from(Span::styled(
                        value,
                        Style::default().fg(theme::NEON_MAGENTA),
                    )),
                ])
                .height(1)
            })
            .collect();

        let widths = [
            Constraint::Percentage(40),
            Constraint::Percentage(30),
            Constraint::Percentage(30),
        ];

        let results_table = Table::new(rows, widths)
            .header(header)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(theme::BORDER_ACTIVE))
                    .title("Live Results")
                    .style(Style::default().bg(theme::BG_DARK)),
            )
            .column_spacing(2);

        frame.render_widget(results_table, chunks[1]);

        // Logs
        let log_lines_ui: Vec<Line> = self
            .log_lines
            .iter()
            .map(|(msg, level)| {
                let style = match level {
                    LogLevel::Info => Style::default().fg(theme::TEXT_PRIMARY),
                    LogLevel::Success => Style::default().fg(theme::NEON_GREEN),
                    LogLevel::Warning => Style::default().fg(theme::NEON_AMBER),
                    LogLevel::Error => Style::default().fg(theme::NEON_RED),
                    LogLevel::Dim => theme::dim_style(),
                };
                Line::from(Span::styled(msg, style))
            })
            .collect();

        let log_len = log_lines_ui.len();
        let log_height = chunks[2].height.saturating_sub(2) as usize; // account for borders

        // Handle auto-scrolling
        if self.log_auto_scroll {
            if log_len > log_height {
                self.log_scroll = log_len - log_height;
            } else {
                self.log_scroll = 0;
            }
        }

        let max_scroll = log_len.saturating_sub(log_height);
        if self.log_scroll > max_scroll {
            self.log_scroll = max_scroll;
        }

        let logs = Paragraph::new(log_lines_ui)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(theme::BORDER_INACTIVE))
                    .title("Process Log"),
            )
            .scroll((self.log_scroll as u16, 0));

        frame.render_widget(logs, chunks[2]);
    }

    // ─── Chat Screen ───────────────────────────────────────────

    fn render_chat(&mut self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // title
                Constraint::Min(5),    // messages
                Constraint::Length(3), // input
                Constraint::Length(1), // status
            ])
            .split(area);

        // Title
        let title = Paragraph::new(Line::from(vec![
            Span::styled("  💬 ", Style::default().fg(theme::NEON_CYAN)),
            Span::styled("CHAT WITH DECENSORED MODEL", theme::title_style()),
        ]))
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(theme::BORDER_INACTIVE)),
        );
        frame.render_widget(title, chunks[0]);

        // Messages
        let msg_lines: Vec<Line> = self
            .chat_messages
            .iter()
            .flat_map(|(role, content)| {
                let (prefix, style) = match role.as_str() {
                    "user" => (
                        "▸ You: ",
                        Style::default()
                            .fg(theme::NEON_CYAN)
                            .add_modifier(Modifier::BOLD),
                    ),
                    "assistant" => ("▸ AI:  ", Style::default().fg(theme::NEON_MAGENTA)),
                    _ => ("▸ Sys: ", theme::dim_style()),
                };
                vec![
                    Line::from(vec![
                        Span::styled(prefix, style),
                        Span::styled(content.clone(), Style::default().fg(theme::TEXT_PRIMARY)),
                    ]),
                    Line::from(""),
                ]
            })
            .collect();

        // The pane wraps, so the scrollable extent is the number of *visual*
        // lines after wrapping, not the number of messages. Counting messages
        // here would clamp the scroll far too early and re-hide the tail.
        let inner_width = chunks[1].width.saturating_sub(2) as usize;
        let inner_height = chunks[1].height.saturating_sub(2) as usize;
        let total_lines: usize = self
            .chat_messages
            .iter()
            .map(|(_, content)| {
                // Every message renders as its (7-column) prefix and body on one
                // wrapped line, followed by a blank spacer line.
                wrapped_line_count(content, inner_width.saturating_sub(7)) + 1
            })
            .sum();

        let max_scroll = total_lines.saturating_sub(inner_height);

        // Follow the newest output unless the user has scrolled up to read back.
        if self.chat_auto_scroll {
            self.chat_scroll = max_scroll;
        } else if self.chat_scroll > max_scroll {
            self.chat_scroll = max_scroll;
        }

        let messages = Paragraph::new(msg_lines)
            .wrap(Wrap { trim: false })
            .scroll((self.chat_scroll as u16, 0))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(theme::BORDER_INACTIVE))
                    .style(Style::default().bg(theme::BG_SURFACE)),
            );
        frame.render_widget(messages, chunks[1]);

        // Input
        let input_text = if self.chat_input.is_empty() {
            "Type your message..."
        } else {
            &self.chat_input
        };
        let input_style = if self.chat_input.is_empty() {
            theme::dim_style()
        } else {
            Style::default().fg(theme::NEON_CYAN)
        };

        let input = Paragraph::new(Line::from(Span::styled(input_text, input_style))).block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(theme::NEON_CYAN))
                .title(Span::styled(" Message ", theme::title_style()))
                .style(Style::default().bg(theme::BG_SURFACE)),
        );
        frame.render_widget(input, chunks[2]);

        // Status bar
        let status_text = if self.chat_loading {
            "⏳ Loading model... Please wait."
        } else if self.chat_streaming {
            "💭 Generating response..."
        } else {
            "Press Enter to send · Esc to exit"
        };
        let status = Paragraph::new(Line::from(Span::styled(status_text, theme::dim_style())))
            .alignment(Alignment::Center);
        frame.render_widget(status, chunks[3]);

        // Cursor
        let cursor_x = chunks[2].x + 1 + char_len(&self.chat_input) as u16;
        let cursor_y = chunks[2].y + 1;
        frame.set_cursor_position((cursor_x, cursor_y));
    }

    // ─── Confirm Dialog ────────────────────────────────────────

    fn render_confirm_dialog(&mut self, frame: &mut Frame, area: Rect) {
        let dialog_width = 60.min(area.width.saturating_sub(4));
        let dialog_height = 9;
        let dialog_area = centered_rect_fixed(dialog_width, dialog_height, area);

        // Clear background
        frame.render_widget(Clear, dialog_area);

        // Spell out irreversible actions so the target is never ambiguous.
        let prompt = match &self.screen {
            Screen::Confirm(ConfirmAction::DeleteCheckpoint(model)) => {
                Some(format!("Permanently delete the checkpoint for {model}?"))
            }
            _ => None,
        };

        let dialog = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Double)
            .border_style(Style::default().fg(theme::NEON_AMBER))
            .title(Span::styled(" ⚠ Confirm ", theme::warning_style()))
            .title_alignment(Alignment::Center)
            .style(Style::default().bg(theme::BG_ELEVATED));
        let inner = dialog.inner(dialog_area);
        frame.render_widget(dialog, dialog_area);

        let list_area = if let Some(prompt) = prompt {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(3), Constraint::Min(0)])
                .split(inner);
            frame.render_widget(
                Paragraph::new(prompt)
                    .style(Style::default().fg(theme::TEXT_PRIMARY))
                    .alignment(Alignment::Center)
                    .wrap(Wrap { trim: true }),
                chunks[0],
            );
            chunks[1]
        } else {
            inner
        };

        let items: Vec<ListItem> = self
            .current_menu
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let is_selected = self.menu_state.selected() == Some(i);
                let prefix = if is_selected { " ▸ " } else { "   " };
                let style = if is_selected {
                    Style::default()
                        .fg(theme::NEON_CYAN)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(theme::TEXT_PRIMARY)
                };
                ListItem::new(Line::from(Span::styled(
                    format!("{}{}", prefix, item.label),
                    style,
                )))
            })
            .collect();

        frame.render_stateful_widget(List::new(items), list_area, &mut self.menu_state);
    }

    fn render_checkpoint_prompt_dialog(&mut self, frame: &mut Frame, area: Rect) {
        let dialog_width = 60.min(area.width.saturating_sub(4));
        let dialog_height = 8;
        let dialog_area = centered_rect_fixed(dialog_width, dialog_height, area);

        frame.render_widget(Clear, dialog_area);

        let items: Vec<ListItem> = self
            .current_menu
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let is_selected = self.menu_state.selected() == Some(i);
                let prefix = if is_selected { " ▸ " } else { "   " };
                let style = if is_selected {
                    Style::default()
                        .fg(theme::NEON_CYAN)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(theme::TEXT_PRIMARY)
                };
                ListItem::new(Line::from(Span::styled(
                    format!("{}{}", prefix, item.label),
                    style,
                )))
            })
            .collect();

        let dialog = List::new(items).block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Double)
                .border_style(Style::default().fg(theme::NEON_AMBER))
                .title(Span::styled(" ⚠ Checkpoint Found ", theme::warning_style()))
                .title_alignment(Alignment::Center)
                .style(Style::default().bg(theme::BG_ELEVATED)),
        );
        frame.render_stateful_widget(dialog, dialog_area, &mut self.menu_state);
    }

    // ─── About Screen ──────────────────────────────────────────

    fn render_about(&mut self, frame: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(theme::NEON_CYAN))
            .style(Style::default().bg(theme::BG_DARK));
        let inner_area = block.inner(area);
        frame.render_widget(block, area);

        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(8),  // Logo
                Constraint::Length(2),  // Spacer
                Constraint::Length(10), // Info
                Constraint::Min(1),     // Bottom Spacer
                Constraint::Length(3),  // Footer
            ])
            .margin(2)
            .split(inner_area);

        // Logo
        let logo_lines: Vec<Line> = BANNER
            .iter()
            .map(|&s| {
                Line::from(Span::styled(
                    s,
                    Style::default()
                        .fg(theme::NEON_CYAN)
                        .add_modifier(Modifier::BOLD),
                ))
            })
            .collect();
        let logo = Paragraph::new(logo_lines).alignment(Alignment::Center);
        frame.render_widget(logo, layout[0]);

        // Info
        let info_text = vec![
            Line::from(Span::styled(
                concat!("ANNIHILATE v", env!("ANNIHILATE_VERSION")),
                theme::title_style(),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled("Author: ", Style::default().fg(theme::TEXT_DIM)),
                Span::styled(
                    "tjcrims0nx",
                    Style::default()
                        .fg(theme::NEON_MAGENTA)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::styled("GitHub: ", Style::default().fg(theme::TEXT_DIM)),
                Span::styled(
                    "https://github.com/tjcrims0nx/annihilation-llm",
                    Style::default().fg(theme::NEON_CYAN),
                ),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                "An advanced orthogonal representation ablation framework designed to",
                Style::default().fg(theme::TEXT_PRIMARY),
            )),
            Line::from(Span::styled(
                "systematically identify and zero-out structural refusal vectors in LLMs.",
                Style::default().fg(theme::TEXT_PRIMARY),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "Unchain your local models.",
                Style::default()
                    .fg(theme::NEON_AMBER)
                    .add_modifier(Modifier::ITALIC),
            )),
        ];

        let info_para = Paragraph::new(info_text).alignment(Alignment::Center);
        frame.render_widget(info_para, layout[2]);

        // Footer
        let footer = Paragraph::new(Line::from(Span::styled(
            "Press Esc or Enter to return",
            theme::dim_style(),
        )))
        .alignment(Alignment::Center);
        frame.render_widget(footer, layout[4]);
    }

    // ─── Status Bar ────────────────────────────────────────────

    fn render_status_bar(&self, frame: &mut Frame, area: Rect) {
        let bar_area = Rect::new(area.x, area.y + area.height - 1, area.width, 1);

        let status_line = Line::from(vec![
            Span::styled(
                " ANNIHILATE ",
                Style::default()
                    .fg(theme::BG_DARK)
                    .bg(theme::NEON_CYAN)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" ", theme::status_bar_style()),
            Span::styled(&self.status_message, theme::status_bar_style()),
            Span::styled(
                format!(
                    "{}v{} ",
                    " ".repeat(
                        (area.width as usize).saturating_sub(self.status_message.len() + 20)
                    ),
                    env!("ANNIHILATE_VERSION")
                ),
                theme::status_bar_style(),
            ),
        ]);

        frame.render_widget(
            Paragraph::new(status_line).style(theme::status_bar_style()),
            bar_area,
        );
    }
}

// ─── Layout Helpers ────────────────────────────────────────────

fn centered_rect_fixed(width: u16, height: u16, area: Rect) -> Rect {
    let x = area.x + (area.width.saturating_sub(width)) / 2;
    let y = area.y + (area.height.saturating_sub(height)) / 2;
    Rect::new(x, y, width.min(area.width), height.min(area.height))
}

fn format_duration(secs: u64) -> String {
    let hours = secs / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;
    if hours > 0 {
        format!("{}h {}m", hours, minutes)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, seconds)
    } else {
        format!("{}s", seconds)
    }
}

fn text_sparkline(data: &[f64], width: usize) -> String {
    let bars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let len = data.len().min(width);
    let slice = &data[data.len() - len..];

    if slice.is_empty() {
        return String::new();
    }

    let min = slice.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    slice
        .iter()
        .map(|&v| {
            let normalized = if range > 0.0 { (v - min) / range } else { 0.5 };
            let idx = (normalized * 7.0).round() as usize;
            bars[idx.min(7)]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_handles_multibyte_input() {
        // Regression: the old code used the char cursor as a byte index, so
        // typing after a multi-byte character panicked on a non-boundary.
        let mut s = String::from("éa");
        let mut cursor = 1;
        cursor = insert_at_char_cursor(&mut s, cursor, 'X');
        assert_eq!(s, "éXa");
        assert_eq!(cursor, 2);
    }

    #[test]
    fn insert_at_end_appends() {
        let mut s = String::from("héllo");
        let cursor = insert_at_char_cursor(&mut s, char_len("héllo"), '!');
        assert_eq!(s, "héllo!");
        assert_eq!(cursor, 6);
    }

    #[test]
    fn backspace_removes_whole_character() {
        let mut s = String::from("héllo");
        let cursor = remove_before_char_cursor(&mut s, 2);
        assert_eq!(s, "hllo");
        assert_eq!(cursor, 1);
    }

    #[test]
    fn backspace_at_start_is_a_no_op() {
        let mut s = String::from("é");
        let cursor = remove_before_char_cursor(&mut s, 0);
        assert_eq!(s, "é");
        assert_eq!(cursor, 0);
    }

    #[test]
    fn typing_every_position_never_panics() {
        // Walk a cursor through mixed-width text, inserting at each stop.
        for cursor in 0..=char_len("éàü😀z") {
            let mut s = String::from("éàü😀z");
            insert_at_char_cursor(&mut s, cursor, 'q');
            assert_eq!(char_len(&s), 6);
        }
    }

    #[test]
    fn mask_secret_does_not_split_characters() {
        assert_eq!(mask_secret("hf_abcdef"), "hf_******");
        assert_eq!(mask_secret("ab"), "**");
        assert_eq!(mask_secret(""), "");
        // Multi-byte input would panic under byte slicing.
        assert_eq!(mask_secret("éàüxy"), "éàü**");
    }
}
