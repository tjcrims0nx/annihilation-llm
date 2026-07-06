//! # Theme Module
//!
//! Defines the neon-dark color palette and style helpers for the Annihilation LLM TUI.
//!
//! Every color is specified as [`Color::Rgb`] so the theme renders consistently
//! on true-color terminals. Helper functions return pre-configured [`Style`]
//! values that keep the rest of the UI code clean and declarative.

use ratatui::style::{Color, Modifier, Style};

// ── Background colors ───────────────────────────────────────────────────────

/// Deep void background used for the root canvas.
pub const BG_DARK: Color = Color::Rgb(10, 10, 15);

/// Slightly lighter surface for panels and cards.
pub const BG_SURFACE: Color = Color::Rgb(18, 18, 31);

/// Elevated surface for popups, modals, and focused widgets.
pub const BG_ELEVATED: Color = Color::Rgb(25, 25, 42);

// ── Neon accent colors ──────────────────────────────────────────────────────

/// Primary neon cyan – the signature accent.
pub const NEON_CYAN: Color = Color::Rgb(0, 255, 240);

/// Hot magenta for secondary accents and highlights.
pub const NEON_MAGENTA: Color = Color::Rgb(255, 0, 255);

/// Electric purple for key hints and decorative elements.
pub const NEON_PURPLE: Color = Color::Rgb(191, 0, 255);

/// Radioactive green for success indicators.
pub const NEON_GREEN: Color = Color::Rgb(57, 255, 20);

/// Warm amber for warnings.
pub const NEON_AMBER: Color = Color::Rgb(255, 170, 0);

/// Neon red for errors and destructive actions.
pub const NEON_RED: Color = Color::Rgb(255, 0, 64);

/// Neon blue for informational elements.
pub const NEON_BLUE: Color = Color::Rgb(0, 120, 255);

// ── Text colors ─────────────────────────────────────────────────────────────

/// Default body text – slightly blue-shifted white.
pub const TEXT_PRIMARY: Color = Color::Rgb(224, 224, 255);

/// De-emphasised / secondary text.
pub const TEXT_DIM: Color = Color::Rgb(106, 106, 138);

/// Full-brightness white for maximum contrast.
pub const TEXT_BRIGHT: Color = Color::Rgb(255, 255, 255);

// ── Border colors ───────────────────────────────────────────────────────────

/// Border for the currently focused widget.
pub const BORDER_ACTIVE: Color = Color::Rgb(0, 204, 204);

/// Border for unfocused widgets.
pub const BORDER_INACTIVE: Color = Color::Rgb(42, 42, 63);

/// Glow-effect border (same hue as [`NEON_CYAN`]).
pub const BORDER_GLOW: Color = Color::Rgb(0, 255, 240);

// ── Style helpers ───────────────────────────────────────────────────────────

/// Bold cyan used for panel titles and section headers.
pub fn title_style() -> Style {
    Style::default().fg(NEON_CYAN).add_modifier(Modifier::BOLD)
}

/// Inverted cyan row style for the currently selected list item.
pub fn selected_style() -> Style {
    Style::default()
        .fg(BG_DARK)
        .bg(NEON_CYAN)
        .add_modifier(Modifier::BOLD)
}

/// Neutral style for non-selected list items.
pub fn unselected_style() -> Style {
    Style::default().fg(TEXT_PRIMARY)
}

/// Subdued style for secondary / decorative text.
pub fn dim_style() -> Style {
    Style::default().fg(TEXT_DIM)
}

/// Bold magenta accent for important labels.
pub fn accent_style() -> Style {
    Style::default()
        .fg(NEON_MAGENTA)
        .add_modifier(Modifier::BOLD)
}

/// Bold green for success messages and completion indicators.
pub fn success_style() -> Style {
    Style::default().fg(NEON_GREEN).add_modifier(Modifier::BOLD)
}

/// Amber style for warning messages.
pub fn warning_style() -> Style {
    Style::default().fg(NEON_AMBER)
}

/// Bold red for error messages and critical alerts.
pub fn error_style() -> Style {
    Style::default().fg(NEON_RED).add_modifier(Modifier::BOLD)
}

/// Dim text on surface background – used for the bottom status bar.
pub fn status_bar_style() -> Style {
    Style::default().fg(TEXT_DIM).bg(BG_SURFACE)
}

/// Returns an active or inactive border style depending on focus state.
pub fn border_style(active: bool) -> Style {
    if active {
        Style::default().fg(BORDER_ACTIVE)
    } else {
        Style::default().fg(BORDER_INACTIVE)
    }
}

/// Cyan-on-elevated style for progress gauges.
pub fn gauge_style() -> Style {
    Style::default().fg(NEON_CYAN).bg(BG_ELEVATED)
}

/// Dim style for scrollback log lines.
pub fn log_style() -> Style {
    Style::default().fg(TEXT_DIM)
}

/// Bold cyan for numeric / important values in status readouts.
pub fn highlight_value() -> Style {
    Style::default().fg(NEON_CYAN).add_modifier(Modifier::BOLD)
}

/// Bold purple for keyboard shortcut indicators.
pub fn key_hint_style() -> Style {
    Style::default()
        .fg(NEON_PURPLE)
        .add_modifier(Modifier::BOLD)
}

/// Dim style for the description text next to key hints.
pub fn key_desc_style() -> Style {
    Style::default().fg(TEXT_DIM)
}
