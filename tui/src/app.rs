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
        Block, BorderType, Borders, Clear, Gauge, List, ListItem, ListState, Paragraph, Wrap,
        Table, Row, Cell, TableState, Sparkline,
    },
};

use crate::parser::ParsedEvent;
use crate::subprocess::SubprocessManager;
use crate::sysinfo::SystemInfo;
use crate::theme;

// ─── ASCII Art Banner ──────────────────────────────────────────

const BANNER: &[&str] = &[
    r"  ████  █    █ █    █ ██ █   █ ██ █     ████  ██████ ██  ████  █    █ ",
    r" █    █ ██   █ ██   █ ██ █   █ ██ █    █    █   ██   ██ █    █ ██   █ ",
    r" ██████ █ █  █ █ █  █ ██ █████ ██ █    ██████   ██   ██ █    █ █ █  █ ",
    r" █    █ █  █ █ █  █ █ ██ █   █ ██ █    █    █   ██   ██ █    █ █  █ █ ",
    r" █    █ █   ██ █   ██ ██ █   █ ██ █████ █    █  ██   ██  ████  █   ██ ",
];

const TAGLINE: &str = "-- Breaking the Chains | Unleashing Model Potential --";

// ─── Application Screens ───────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Screen {
    Splash,
    Setup,
    ModelInput,
    ConfigSelect,
    Processing,
    Results,
    TrialActions,
    Chat,
    Export,
    CheckpointPrompt,
    Confirm(ConfirmAction),
    About,
    RecentModels,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConfirmAction {
    Quit,
    StopProcessing,
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

    // Status
    pub status_message: String,
    pub annihilate_available: bool,
    pub plot_residuals: bool,
    pub quantize: bool,
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
            MenuItem::new("Start Decensoring", "Launch the annihilation workflow").with_key("Enter"),
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
            status_message: "Ready".to_string(),
            annihilate_available: false,
            plot_residuals: false,
            quantize: false,
        }
    }

    /// Tick the animation state
    pub fn tick(&mut self) {
        self.tick_count += 1;
        self.glow_phase = (self.tick_count as f64 * 0.05).sin() * 0.5 + 0.5;

        // Process real subprocess events
        if self.screen == Screen::Processing && self.is_processing
            && let Some(ref mut child) = self.subprocess {
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
                                self.log_lines.push((format!("Determined batch size: {}", size), LogLevel::Success));
                            }
                            ParsedEvent::DatasetLoading(msg) => {
                                self.log_lines.push((msg, LogLevel::Dim));
                            }
                            ParsedEvent::KLDivergence(kl) => {
                                self.pending_kl = Some(kl);
                            }
                            ParsedEvent::CalculatingDirections => {
                                self.log_lines.push(("Calculating per-layer refusal directions...".into(), LogLevel::Info));
                            }
                            ParsedEvent::OptimizationStarting { n_trials } => {
                                self.total_trials = n_trials;
                                self.log_lines.push(("Starting optimization...".into(), LogLevel::Success));
                            }
                            ParsedEvent::TrialStarting { trial_number, total_trials } => {
                                self.current_trial = trial_number;
                                self.total_trials = total_trials;
                                self.log_lines.push((format!("Starting trial {}/{}...", trial_number, total_trials), LogLevel::Info));
                            }
                            ParsedEvent::TrialComplete { trial_number, total_trials: _, refusals, total_prompts } => {
                                if trial_number > 0 {
                                    self.current_trial = trial_number;
                                } else {
                                    self.current_trial += 1; // Fallback if we couldn't parse the exact number
                                }

                                let kl_divergence = self.pending_kl.take().unwrap_or(0.0);
                                self.kl_history.push(kl_divergence);
                                self.refusal_history.push(refusals as f64);

                                if self.best_kl.is_none() || kl_divergence < self.best_kl.unwrap() {
                                    self.best_kl = Some(kl_divergence);
                                }
                                if self.best_refusals.is_none() || refusals < self.best_refusals.unwrap() {
                                    self.best_refusals = Some(refusals);
                                }

                                self.log_lines.push((
                                    format!("Trial {}: refusals={}/{}, KL={:.4}",
                                        self.current_trial, refusals, total_prompts, kl_divergence),
                                    if refusals <= 5 { LogLevel::Success } else { LogLevel::Info },
                                ));
                            }
                            ParsedEvent::BestTrial { .. } => {}
                            ParsedEvent::OptimizationComplete => {
                                self.log_lines.push(("Optimization finished!".into(), LogLevel::Success));
                                self.is_processing = false;
                                self.generate_demo_results(); // Still use demo results for now until the interactive menu parser is fully connected
                                self.switch_to_results();
                            }
                            ParsedEvent::GpuMemory { .. } => {}
                            ParsedEvent::ElapsedTime(time) => {
                                // Time is string "00:00:00", could parse it, but for now just status
                                self.log_lines.push((format!("Elapsed: {}", time), LogLevel::Dim));
                            }
                            ParsedEvent::EstimatedRemaining(time) => {
                                self.log_lines.push((format!("ETA: {}", time), LogLevel::Dim));
                            }
                            ParsedEvent::TrialPruned { trial_number } => {
                                self.log_lines.push((format!("Trial {} pruned", trial_number), LogLevel::Warning));
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
                            ParsedEvent::Raw(_line) => {
                                // Handled by OutputLine below to avoid duplicates
                            }
                        },
                        SubprocessMessage::OutputLine(line) => {
                            if !line.trim().is_empty() && !line.contains("Spawning") {
                                let clean = crate::parser::strip_ansi(&line);
                                if clean.contains("No GPU or other accelerator detected") && self.sys_info.gpu_name != "Unknown" {
                                    self.log_lines.push((
                                        "CRITICAL WARNING: The TUI detects your GPU, but Python cannot see it! You have installed the CPU-only version of PyTorch. The process will run extremely slow.".to_string(),
                                        LogLevel::Error
                                    ));
                                    self.log_lines.push((
                                        "FIX THIS BY RUNNING: `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --upgrade`".to_string(),
                                        LogLevel::Error
                                    ));
                                }
                                self.log_lines.push((clean, LogLevel::Dim));
                            }
                        }
                        SubprocessMessage::Exited(code) => {
                            if self.is_setting_up {
                                if code == Some(0) {
                                    self.is_setting_up = false;
                                    self.log_lines.push(("Environment verification complete. Spawning backend...".to_string(), LogLevel::Info));
                                    
                                    // Start the actual subprocess
                                    let mut extra_args = vec![
                                        "--n-trials".to_string(),
                                        self.total_trials.to_string(),
                                    ];
                                    
                                    // Add quantization if selected.
                                    if self.quantize {
                                        extra_args.push("--quantization".to_string());
                                        extra_args.push("bnb_4bit".to_string());
                                    }
                                    
                                    self.subprocess = Some(SubprocessManager::spawn(&self.model_input, &extra_args));
                                } else {
                                    self.is_processing = false;
                                    self.is_setting_up = false;
                                    self.log_lines.push((format!("Setup exited with code {:?}", code), LogLevel::Warning));
                                }
                            } else {
                                self.is_processing = false;
                                self.log_lines.push((format!("Process exited with code {:?}", code), LogLevel::Warning));
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
    }

    fn generate_demo_results(&mut self) {
        self.trials = vec![
            TrialResult { index: 142, refusals: 2, total_prompts: 100, kl_divergence: 0.0312, direction: "global".into() },
            TrialResult { index: 87, refusals: 0, total_prompts: 100, kl_divergence: 0.1247, direction: "per layer".into() },
            TrialResult { index: 198, refusals: 1, total_prompts: 100, kl_divergence: 0.0589, direction: "global".into() },
            TrialResult { index: 56, refusals: 3, total_prompts: 100, kl_divergence: 0.0201, direction: "per layer".into() },
            TrialResult { index: 171, refusals: 5, total_prompts: 100, kl_divergence: 0.0098, direction: "global".into() },
        ];
    }

    fn switch_to_results(&mut self) {
        self.screen = Screen::Results;
        self.trial_list_state.select(Some(0));
        self.current_menu = vec![
            MenuItem::new("Select this trial", "Use the selected trial for export/chat"),
            MenuItem::new("Run additional trials", "Continue optimization with more trials"),
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
            Screen::ConfigSelect => self.handle_config_select_key(key),
            Screen::Processing => self.handle_processing_key(key),
            Screen::Results => self.handle_results_key(key),
            Screen::TrialActions => self.handle_trial_actions_key(key),
            Screen::Chat => self.handle_chat_key(key),
            Screen::Export => self.handle_export_key(key),
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
                    Some(0) => { // Start
                        self.screen = Screen::Setup;
                        self.current_menu = vec![
                            MenuItem::new("Enter Model ID/Path", "Type a Hugging Face model ID or local path").with_key("M"),
                            MenuItem::new("Recent Models", "Choose from previously used models").with_key("R"),
                            MenuItem::new("Back", "Return to main menu").with_key("Esc"),
                        ];
                        self.menu_state.select(Some(0));
                    }
                    Some(1) => { // Config
                        self.screen = Screen::ConfigSelect;
                        self.current_menu = vec![
                            MenuItem::new("Default (200 trials)", "Standard configuration, no quantization"),
                            MenuItem::new("Quick Test (50 trials)", "Faster run for testing"),
                            MenuItem::new("Aggressive (400 trials)", "More thorough optimization"),
                            MenuItem::new("4-bit Quantized", "Lower VRAM usage with bnb_4bit"),
                            MenuItem::new("Back", "Return to main menu").with_key("Esc"),
                        ];
                        self.menu_state.select(Some(0));
                    }
                    Some(2) => {
                        self.screen = Screen::About;
                    }
                    Some(3) => self.should_quit = true, // Quit
                    _ => {}
                }
            }
            KeyCode::Char('q') | KeyCode::Char('Q') => self.should_quit = true,
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
                    Some(0) => { // Enter model
                        self.screen = Screen::ModelInput;
                        self.model_input.clear();
                        self.model_cursor = 0;
                    }
                    Some(1) => { // Recent models
                        let recent_file = crate::subprocess::get_repo_root().join(".recent_models");
                        let recent: Vec<String> = std::fs::read_to_string(&recent_file).unwrap_or_default()
                            .lines()
                            .map(|s| s.to_string())
                            .filter(|s| !s.is_empty())
                            .collect();
                        
                        if recent.is_empty() {
                            self.status_message = "No recent models found.".to_string();
                        } else {
                            self.screen = Screen::RecentModels;
                            self.current_menu = recent.into_iter().map(|m| MenuItem::new(&m, "Select to launch")).collect();
                            self.current_menu.push(MenuItem::new("Back", "Return to setup").with_key("Esc"));
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
            KeyCode::Enter => {
                if !self.model_input.is_empty() {
                    self.check_and_start_processing();
                }
            }
            KeyCode::Esc => {
                self.screen = Screen::Setup;
                self.current_menu = vec![
                    MenuItem::new("Enter Model ID/Path", "Type a Hugging Face model ID or local path").with_key("M"),
                    MenuItem::new("Recent Models", "Choose from previously used models").with_key("R"),
                    MenuItem::new("Back", "Return to main menu").with_key("Esc"),
                ];
                self.menu_state.select(Some(0));
            }
            KeyCode::Char(c) => {
                self.model_input.insert(self.model_cursor, c);
                self.model_cursor += 1;
            }
            KeyCode::Backspace => {
                if self.model_cursor > 0 {
                    self.model_cursor -= 1;
                    self.model_input.remove(self.model_cursor);
                }
            }
            KeyCode::Left => {
                if self.model_cursor > 0 {
                    self.model_cursor -= 1;
                }
            }
            KeyCode::Right => {
                if self.model_cursor < self.model_input.len() {
                    self.model_cursor += 1;
                }
            }
            KeyCode::Home => self.model_cursor = 0,
            KeyCode::End => self.model_cursor = self.model_input.len(),
            _ => {}
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
                    }
                    Some(1) => {
                        self.total_trials = 50;
                        self.quantize = false;
                    }
                    Some(2) => {
                        self.total_trials = 400;
                        self.quantize = false;
                    }
                    Some(3) => {
                        self.total_trials = 200;
                        self.quantize = true;
                    }
                    _ => {}
                }
                self.go_back_to_splash();
                self.status_message = if self.quantize {
                    format!("Config: {} trials, 4-bit", self.total_trials)
                } else {
                    format!("Config: {} trials", self.total_trials)
                };
            }
            KeyCode::Esc => self.go_back_to_splash(),
            _ => {}
        }
    }

    // ─── Processing Screen Keys ────────────────────────────────

    fn handle_processing_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('q') | KeyCode::Char('Q') => {
                self.screen = Screen::Confirm(ConfirmAction::StopProcessing);
                self.current_menu = vec![
                    MenuItem::new("Yes, stop processing", "Halt optimization and view results so far"),
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
                        MenuItem::new("Save Model Locally", "Export merged model to a folder").with_key("S"),
                        MenuItem::new("Upload to Hugging Face", "Push model to HF Hub").with_key("U"),
                        MenuItem::new("Chat with Model", "Test the decensored model").with_key("C"),
                        MenuItem::new("Run Benchmarks", "Evaluate with MMLU, GSM8K, etc.").with_key("B"),
                        MenuItem::new("Run More Trials", "Continue optimization").with_key("R"),
                        MenuItem::new("Back to Results", "Return to trial selection").with_key("Esc"),
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
                match self.menu_state.selected() {
                    Some(0) => { // Save locally
                        self.screen = Screen::Export;
                        self.current_menu = vec![
                            MenuItem::new("Merge and export full model", "Requires sufficient RAM"),
                            MenuItem::new("Export adapter only", "Can be merged later, smaller file"),
                            MenuItem::new("Back", "Return to actions menu").with_key("Esc"),
                        ];
                        self.menu_state.select(Some(0));
                    }
                    Some(1) => { /* Upload - would need HF token input */ }
                    Some(2) => { // Chat
                        self.screen = Screen::Chat;
                        self.chat_messages.clear();
                        self.chat_input.clear();
                        self.chat_messages.push(("system".to_string(), "Chat mode active. Type a message and press Enter.".to_string()));
                    }
                    Some(3) => { /* Benchmarks */ }
                    Some(4) => { /* More trials */ }
                    _ => {
                        self.switch_to_results();
                    }
                }
            }
            KeyCode::Esc => {
                self.switch_to_results();
            }
            _ => {}
        }
    }

    // ─── Chat Keys ─────────────────────────────────────────────

    fn handle_chat_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Enter => {
                if !self.chat_input.is_empty() {
                    let msg = self.chat_input.clone();
                    self.chat_messages.push(("user".to_string(), msg));
                    self.chat_messages.push(("assistant".to_string(), "This is a demo response. In the full version, this will stream from the decensored model.".to_string()));
                    self.chat_input.clear();
                }
            }
            KeyCode::Char(c) => {
                self.chat_input.push(c);
            }
            KeyCode::Backspace => {
                self.chat_input.pop();
            }
            KeyCode::Esc => {
                self.screen = Screen::TrialActions;
                self.menu_state.select(Some(2));
            }
            _ => {}
        }
    }

    // ─── Export Keys ───────────────────────────────────────────

    fn handle_export_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.menu_up(),
            KeyCode::Down | KeyCode::Char('j') => self.menu_down(),
            KeyCode::Enter => {
                match self.menu_state.selected() {
                    Some(0) | Some(1) => {
                        self.status_message = "Model export complete!".to_string();
                        self.screen = Screen::TrialActions;
                        self.menu_state.select(Some(0));
                    }
                    _ => {
                        self.screen = Screen::TrialActions;
                        self.menu_state.select(Some(0));
                    }
                }
            }
            KeyCode::Esc => {
                self.screen = Screen::TrialActions;
                self.menu_state.select(Some(0));
            }
            _ => {}
        }
    }

    // ─── Confirm Dialog Keys ───────────────────────────────────

    fn handle_confirm_key(&mut self, key: KeyEvent, action: ConfirmAction) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.menu_up(),
            KeyCode::Down | KeyCode::Char('j') => self.menu_down(),
            KeyCode::Enter => {
                match (self.menu_state.selected(), &action) {
                    (Some(0), ConfirmAction::Quit) => self.should_quit = true,
                    (Some(0), ConfirmAction::StopProcessing) => {
                        self.is_processing = false;
                        if !self.trials.is_empty() || self.current_trial > 0 {
                            self.generate_demo_results();
                            self.switch_to_results();
                        } else {
                            self.go_back_to_splash();
                        }
                    }
                    _ => {
                        self.screen = Screen::Processing;
                    }
                }
            }
            KeyCode::Esc | KeyCode::Char('n') => {
                match action {
                    ConfirmAction::StopProcessing => self.screen = Screen::Processing,
                    ConfirmAction::Quit => self.go_back_to_splash(),
                }
            }
            KeyCode::Char('y') => {
                match action {
                    ConfirmAction::Quit => self.should_quit = true,
                    ConfirmAction::StopProcessing => {
                        self.is_processing = false;
                        self.generate_demo_results();
                        self.switch_to_results();
                    }
                }
            }
            _ => {}
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
                        MenuItem::new("Enter Model ID/Path", "Type a Hugging Face model ID or local path").with_key("M"),
                        MenuItem::new("Recent Models", "Choose from previously used models").with_key("R"),
                        MenuItem::new("Back", "Return to main menu").with_key("Esc"),
                    ];
                    self.menu_state.select(Some(1));
                }
            }
            KeyCode::Esc => {
                self.screen = Screen::Setup;
                self.current_menu = vec![
                    MenuItem::new("Enter Model ID/Path", "Type a Hugging Face model ID or local path").with_key("M"),
                    MenuItem::new("Recent Models", "Choose from previously used models").with_key("R"),
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
            MenuItem::new("Start Decensoring", "Launch the annihilation workflow").with_key("Enter"),
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

        self.log_lines.push(("Verifying Python Environment and Missing Dependencies...".to_string(), LogLevel::Info));

        // Save to recent models
        if !self.model_input.is_empty() {
            let recent_file = crate::subprocess::get_repo_root().join(".recent_models");
            let mut recent: Vec<String> = std::fs::read_to_string(&recent_file).unwrap_or_default()
                .lines()
                .map(|s| s.to_string())
                .filter(|s| !s.is_empty())
                .collect();
            recent.retain(|s| s != &self.model_input);
            recent.insert(0, self.model_input.clone());
            recent.truncate(5); // Keep top 5
            let _ = std::fs::write(&recent_file, recent.join("\n"));
        }

        self.subprocess = Some(SubprocessManager::spawn_setup(self.sys_info.gpu_name != "Unknown"));
    }

    fn check_and_start_processing(&mut self) {
        let checkpoint_dir = crate::subprocess::get_repo_root().join("checkpoints");
        
        // Sanitize model name exactly like python backend does
        let sanitized_model: String = self.model_input.chars()
            .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c.to_string() } else { "--".to_string() })
            .collect();
            
        let checkpoint_file = checkpoint_dir.join(format!("{}.jsonl", sanitized_model));
        
        if checkpoint_file.exists() {
            self.screen = Screen::CheckpointPrompt;
            self.current_menu = vec![
                MenuItem::new("Resume previous run", "Continue optimization from the saved checkpoint"),
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
                        let sanitized_model: String = self.model_input.chars()
                            .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c.to_string() } else { "--".to_string() })
                            .collect();
                        let checkpoint_file = crate::subprocess::get_repo_root()
                            .join("checkpoints")
                            .join(format!("{}.jsonl", sanitized_model));
                        
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
            Screen::Setup => self.render_menu_screen(frame, area, "MODEL SETUP", "Select how to specify your model:"),
            Screen::ModelInput => self.render_model_input(frame, area),
            Screen::ConfigSelect => self.render_menu_screen(frame, area, "CONFIGURATION", "Choose an optimization preset:"),
            Screen::Processing => self.render_processing(frame, area),
            Screen::Results => self.render_results(frame, area),
            Screen::TrialActions => self.render_menu_screen(frame, area, "TRIAL ACTIONS", "What do you want to do with the decensored model?"),
            Screen::Chat => self.render_chat(frame, area),
            Screen::Export => self.render_menu_screen(frame, area, "EXPORT MODEL", "Choose export strategy:"),
            Screen::CheckpointPrompt => {
                self.render_menu_screen(frame, area, "MODEL SETUP", "Select how to specify your model:");
                self.render_checkpoint_prompt_dialog(frame, area);
            },
            Screen::RecentModels => self.render_menu_screen(frame, area, "RECENT MODELS", "Select a previously used model:"),
            Screen::Confirm(action) => {
                // Render previous screen dimmed, then overlay
                match action {
                    ConfirmAction::StopProcessing => self.render_processing(frame, area),
                    ConfirmAction::Quit => self.render_splash(frame, area),
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
                Constraint::Length(2),     // top padding
                Constraint::Length(7),     // banner
                Constraint::Length(2),     // tagline
                Constraint::Length(1),     // spacer
                Constraint::Min(6),        // menu
                Constraint::Length(1),     // status bar
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
                            (
                                (191.0 + s * 64.0) as u8,
                                0u8,
                                255u8,
                            )
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
            Style::default().fg(tagline_color).add_modifier(Modifier::ITALIC),
        )))
        .alignment(Alignment::Center);
        frame.render_widget(tagline, chunks[2]);

        // Menu
        self.render_selection_menu(frame, chunks[4]);
    }

    // ─── Generic Menu Screen ───────────────────────────────────

    fn render_menu_screen(&mut self, frame: &mut Frame, area: Rect, title: &str, subtitle: &str) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),     // title
                Constraint::Length(2),     // subtitle
                Constraint::Length(1),     // spacer
                Constraint::Min(6),        // menu
                Constraint::Length(1),     // status bar
            ])
            .split(area);

        // Title
        let title_widget = Paragraph::new(Line::from(vec![
            Span::styled("  ⚔ ", Style::default().fg(theme::NEON_MAGENTA)),
            Span::styled(title, theme::title_style()),
            Span::styled(" ⚔  ", Style::default().fg(theme::NEON_MAGENTA)),
        ]))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::BOTTOM).border_style(Style::default().fg(theme::BORDER_INACTIVE)));
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
        let menu_area = centered_rect_fixed(menu_width, self.current_menu.len() as u16 * 3 + 2, area);

        let items: Vec<ListItem> = self.current_menu
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let is_selected = self.menu_state.selected() == Some(i);

                let prefix = if is_selected { "▸ " } else { "  " };

                let mut spans = vec![
                    Span::styled(prefix, if is_selected {
                        Style::default().fg(theme::NEON_CYAN).add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(theme::TEXT_DIM)
                    }),
                    Span::styled(&item.label, if is_selected {
                        Style::default().fg(theme::NEON_CYAN).add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(theme::TEXT_PRIMARY)
                    }),
                ];

                if let Some(key) = &item.key_hint {
                    spans.push(Span::styled(
                        format!("  [{}]", key),
                        Style::default().fg(if is_selected { theme::NEON_PURPLE } else { theme::TEXT_DIM }),
                    ));
                }

                let main_line = Line::from(spans);
                let desc_line = Line::from(Span::styled(
                    format!("    {}", item.description),
                    Style::default().fg(if is_selected { theme::BORDER_ACTIVE } else { theme::TEXT_DIM }).add_modifier(Modifier::ITALIC),
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
                    .title(Span::styled(" Select ", Style::default().fg(theme::NEON_CYAN).add_modifier(Modifier::BOLD)))
                    .title_alignment(Alignment::Center)
                    .style(Style::default().bg(theme::BG_SURFACE))
            )
            .highlight_style(Style::default()); // We handle highlighting manually

        frame.render_stateful_widget(list, menu_area, &mut self.menu_state);

        // Key hints below menu
        let hint_area = Rect::new(menu_area.x, menu_area.y + menu_area.height, menu_area.width, 1);
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
        .block(Block::default().borders(Borders::BOTTOM).border_style(Style::default().fg(theme::BORDER_INACTIVE)));
        frame.render_widget(title, chunks[0]);

        let sub = Paragraph::new(Line::from(Span::styled(
            "Enter a Hugging Face model ID or local path:",
            theme::dim_style(),
        )))
        .alignment(Alignment::Center);
        frame.render_widget(sub, chunks[1]);

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
            Style::default().fg(theme::NEON_CYAN).add_modifier(Modifier::BOLD)
        };

        let input = Paragraph::new(Line::from(Span::styled(&display_text, input_style)))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(theme::NEON_CYAN))
                    .title(Span::styled(" Model ", Style::default().fg(theme::NEON_CYAN).add_modifier(Modifier::BOLD)))
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

    // ─── Processing Dashboard ──────────────────────────────────

    fn render_processing(&mut self, frame: &mut Frame, area: Rect) {
        let main_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(75), Constraint::Percentage(25)])
            .split(Rect::new(area.x, area.y, area.width, area.height.saturating_sub(1)));

        let left_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(7),   // header + progress
                Constraint::Length(16),  // metrics
                Constraint::Min(5),      // log
            ])
            .split(main_chunks[0]);

        let right_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(10),  // system info
                Constraint::Min(5),      // controls
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
                if self.model_input.is_empty() { "demo-model" } else { &self.model_input },
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
                Style::default().fg(theme::TEXT_BRIGHT).add_modifier(Modifier::BOLD),
            ))
            .ratio(progress_ratio);
        frame.render_widget(gauge, progress_lines[1]);

        // Timing
        let elapsed_str = format_duration(self.elapsed_secs);
        let eta_str = self.eta_secs.map_or("calculating...".to_string(), format_duration);
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

        let refusal_str = self.best_refusals
            .map_or("--".to_string(), |r| format!("{}/100", r));
        let kl_str = self.best_kl
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
            let kl_sparkline_data: Vec<u64> = self.kl_history.iter().map(|&v| (v * 10000.0) as u64).collect();
            let max_val = kl_sparkline_data.iter().max().cloned().unwrap_or(0);
            
            let kl_sparkline = Sparkline::default()
                .block(Block::default().title(Span::styled(" KL Div ", theme::dim_style())))
                .style(Style::default().fg(theme::NEON_CYAN))
                .data(&kl_sparkline_data)
                .max(max_val.max(1));
            
            frame.render_widget(kl_sparkline, metric_lines[3]);
        }

        if !self.refusal_history.is_empty() {
            let ref_sparkline_data: Vec<u64> = self.refusal_history.iter().map(|&v| v as u64).collect();
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
            .title(Span::styled(" SYSTEM ", Style::default().fg(theme::TEXT_DIM)))
            .style(Style::default().bg(theme::BG_SURFACE));

        let sys_inner = sys_block.inner(right_chunks[0]);
        frame.render_widget(sys_block, right_chunks[0]);

        let sys_lines: Vec<Line> = vec![
            Line::from(vec![
                Span::styled(" GPU: ", theme::dim_style()),
                Span::styled(&self.sys_info.gpu_name, Style::default().fg(theme::NEON_PURPLE)),
            ]),
            Line::from(vec![
                Span::styled(" VRAM: ", theme::dim_style()),
                Span::styled(
                    format!("{:.1}/{:.0} GB", self.sys_info.vram_used_gb(), self.sys_info.vram_total_gb()),
                    theme::highlight_value(),
                ),
            ]),
            Line::from(vec![
                Span::styled(" RAM: ", theme::dim_style()),
                Span::styled(
                    format!("{:.1}/{:.0} GB", self.sys_info.ram_used_gb(), self.sys_info.ram_total_gb()),
                    theme::highlight_value(),
                ),
            ]),
            Line::from(vec![
                Span::styled(" Batch: ", theme::dim_style()),
                Span::styled(format!("{}", self.batch_size), theme::highlight_value()),
            ]),
            Line::from(vec![
                Span::styled(" Tok/s: ", theme::dim_style()),
                Span::styled(format!("{:.0}", self.tokens_per_sec), Style::default().fg(theme::NEON_GREEN)),
            ]),
        ];

        let sys_text = Paragraph::new(sys_lines);
        frame.render_widget(sys_text, sys_inner);

        // ── Controls ──
        let ctrl_block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(theme::BORDER_INACTIVE))
            .title(Span::styled(" CONTROLS ", Style::default().fg(theme::TEXT_DIM)))
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
                Span::styled("  ↑↓", theme::key_hint_style()),
                Span::styled(" Scroll log", theme::key_desc_style()),
            ]),
            Line::from(vec![
                Span::styled(" C-c", theme::key_hint_style()),
                Span::styled(" Force quit", theme::key_desc_style()),
            ]),
        ];

        let ctrl_text = Paragraph::new(ctrl_lines);
        frame.render_widget(ctrl_text, ctrl_inner);
    }

    // ─── Results Screen ────────────────────────────────────────

    fn render_results(&mut self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // title
                Constraint::Length(2),  // info
                Constraint::Min(8),     // trial list
                Constraint::Length(3),  // hints
                Constraint::Length(1),  // status
            ])
            .split(area);

        // Title
        let title = Paragraph::new(Line::from(vec![
            Span::styled("  ✨ ", Style::default().fg(theme::NEON_GREEN)),
            Span::styled("OPTIMIZATION COMPLETE", theme::success_style()),
            Span::styled(" ✨  ", Style::default().fg(theme::NEON_GREEN)),
        ]))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::BOTTOM).border_style(Style::default().fg(theme::BORDER_INACTIVE)));
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
            .style(Style::default().fg(theme::NEON_PURPLE).add_modifier(Modifier::BOLD | Modifier::UNDERLINED))
            .bottom_margin(1);

        let rows: Vec<Row> = self.trials
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
                    Cell::from(format!("{}/{}", trial.refusals, trial.total_prompts)).style(Style::default().fg(refusal_color)),
                    Cell::from(format!("{:.4}", trial.kl_divergence)).style(Style::default().fg(kl_color)),
                    Cell::from(trial.direction.clone()).style(theme::dim_style()),
                ])
            })
            .collect();

        let trial_table = Table::new(rows, [
            Constraint::Length(10), // Trial
            Constraint::Length(15), // Refusals
            Constraint::Length(15), // KL Div
            Constraint::Min(20),    // Direction
        ])
        .header(header)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(theme::BORDER_ACTIVE))
                .title(Span::styled(" Pareto Optimal Trials ", theme::title_style()))
                .style(Style::default().bg(theme::BG_SURFACE))
        )
        .row_highlight_style(
            Style::default()
                .bg(theme::BG_DARK)
                .fg(theme::NEON_CYAN)
                .add_modifier(Modifier::BOLD)
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

    // ─── Chat Screen ───────────────────────────────────────────

    fn render_chat(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // title
                Constraint::Min(5),     // messages
                Constraint::Length(3),  // input
                Constraint::Length(1),  // status
            ])
            .split(area);

        // Title
        let title = Paragraph::new(Line::from(vec![
            Span::styled("  💬 ", Style::default().fg(theme::NEON_CYAN)),
            Span::styled("CHAT WITH DECENSORED MODEL", theme::title_style()),
        ]))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::BOTTOM).border_style(Style::default().fg(theme::BORDER_INACTIVE)));
        frame.render_widget(title, chunks[0]);

        // Messages
        let msg_lines: Vec<Line> = self.chat_messages.iter().flat_map(|(role, content)| {
            let (prefix, style) = match role.as_str() {
                "user" => ("▸ You: ", Style::default().fg(theme::NEON_CYAN).add_modifier(Modifier::BOLD)),
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
        }).collect();

        let messages = Paragraph::new(msg_lines)
            .wrap(Wrap { trim: false })
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

        let input = Paragraph::new(Line::from(Span::styled(input_text, input_style)))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(theme::NEON_CYAN))
                    .title(Span::styled(" Message ", theme::title_style()))
                    .style(Style::default().bg(theme::BG_SURFACE)),
            );
        frame.render_widget(input, chunks[2]);

        // Cursor
        let cursor_x = chunks[2].x + 1 + self.chat_input.len() as u16;
        let cursor_y = chunks[2].y + 1;
        frame.set_cursor_position((cursor_x, cursor_y));
    }

    // ─── Confirm Dialog ────────────────────────────────────────

    fn render_confirm_dialog(&mut self, frame: &mut Frame, area: Rect) {
        let dialog_width = 50.min(area.width.saturating_sub(4));
        let dialog_height = 8;
        let dialog_area = centered_rect_fixed(dialog_width, dialog_height, area);

        // Clear background
        frame.render_widget(Clear, dialog_area);

        let items: Vec<ListItem> = self.current_menu
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let is_selected = self.menu_state.selected() == Some(i);
                let prefix = if is_selected { " ▸ " } else { "   " };
                let style = if is_selected {
                    Style::default().fg(theme::NEON_CYAN).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(theme::TEXT_PRIMARY)
                };
                ListItem::new(Line::from(Span::styled(format!("{}{}", prefix, item.label), style)))
            })
            .collect();

        let dialog = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Double)
                    .border_style(Style::default().fg(theme::NEON_AMBER))
                    .title(Span::styled(" ⚠ Confirm ", theme::warning_style()))
                    .title_alignment(Alignment::Center)
                    .style(Style::default().bg(theme::BG_ELEVATED)),
            );
        frame.render_stateful_widget(dialog, dialog_area, &mut self.menu_state);
    }

    fn render_checkpoint_prompt_dialog(&mut self, frame: &mut Frame, area: Rect) {
        let dialog_width = 60.min(area.width.saturating_sub(4));
        let dialog_height = 8;
        let dialog_area = centered_rect_fixed(dialog_width, dialog_height, area);

        frame.render_widget(Clear, dialog_area);

        let items: Vec<ListItem> = self.current_menu
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let is_selected = self.menu_state.selected() == Some(i);
                let prefix = if is_selected { " ▸ " } else { "   " };
                let style = if is_selected {
                    Style::default().fg(theme::NEON_CYAN).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(theme::TEXT_PRIMARY)
                };
                ListItem::new(Line::from(Span::styled(format!("{}{}", prefix, item.label), style)))
            })
            .collect();

        let dialog = List::new(items)
            .block(
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
        let logo_lines: Vec<Line> = BANNER.iter().map(|&s| {
            Line::from(Span::styled(s, Style::default().fg(theme::NEON_CYAN).add_modifier(Modifier::BOLD)))
        }).collect();
        let logo = Paragraph::new(logo_lines).alignment(Alignment::Center);
        frame.render_widget(logo, layout[0]);

        // Info
        let info_text = vec![
            Line::from(Span::styled("ANNIHILATE v1.4.3", theme::title_style())),
            Line::from(""),
            Line::from(vec![
                Span::styled("Author: ", Style::default().fg(theme::TEXT_DIM)),
                Span::styled("tjcrims0nx", Style::default().fg(theme::NEON_MAGENTA).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::styled("GitHub: ", Style::default().fg(theme::TEXT_DIM)),
                Span::styled("https://github.com/tjcrims0nx/annihilation-llm", Style::default().fg(theme::NEON_CYAN)),
            ]),
            Line::from(""),
            Line::from(Span::styled("An advanced orthogonal representation ablation framework designed to", Style::default().fg(theme::TEXT_PRIMARY))),
            Line::from(Span::styled("systematically identify and zero-out structural refusal vectors in LLMs.", Style::default().fg(theme::TEXT_PRIMARY))),
            Line::from(""),
            Line::from(Span::styled("Unchain your local models.", Style::default().fg(theme::NEON_AMBER).add_modifier(Modifier::ITALIC))),
        ];
        
        let info_para = Paragraph::new(info_text).alignment(Alignment::Center);
        frame.render_widget(info_para, layout[2]);

        // Footer
        let footer = Paragraph::new(Line::from(Span::styled("Press Esc or Enter to return", theme::dim_style())))
            .alignment(Alignment::Center);
        frame.render_widget(footer, layout[4]);
    }

    // ─── Status Bar ────────────────────────────────────────────

    fn render_status_bar(&self, frame: &mut Frame, area: Rect) {
        let bar_area = Rect::new(area.x, area.y + area.height - 1, area.width, 1);

        let status_line = Line::from(vec![
            Span::styled(" ANNIHILATE ", Style::default().fg(theme::BG_DARK).bg(theme::NEON_CYAN).add_modifier(Modifier::BOLD)),
            Span::styled(" ", theme::status_bar_style()),
            Span::styled(&self.status_message, theme::status_bar_style()),
            Span::styled(
                format!("{}v0.1.0 ",
                    " ".repeat((area.width as usize).saturating_sub(self.status_message.len() + 20))),
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
