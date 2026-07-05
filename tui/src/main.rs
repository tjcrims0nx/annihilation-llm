#![allow(dead_code)]
//! ANNTUI - Beautiful neon TUI for Annihilation LLM
//!
//! A user-friendly terminal interface with selection menus,
//! live dashboards, and dark neon aesthetics.

mod app;
mod events;
mod parser;
mod subprocess;
mod sysinfo;
mod theme;

use std::io;

use app::App;
use color_eyre::Result;
use crossterm::{
    ExecutableCommand,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use events::{AppEvent, EventHandler};
use ratatui::prelude::*;

fn main() -> Result<()> {
    // Install color-eyre panic/error hooks
    color_eyre::install()?;

    // Setup terminal - ensure a completely fresh screen
    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    io::stdout().execute(crossterm::cursor::Hide)?;
    io::stdout().execute(crossterm::terminal::Clear(crossterm::terminal::ClearType::All))?;

    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    // Run the app
    let result = run_app(&mut terminal);

    // Restore terminal
    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;
    io::stdout().execute(crossterm::cursor::Show)?;

    result
}

fn run_app(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    let mut app = App::new();
    let events = EventHandler::new(33); // ~30fps

    loop {
        // Draw the UI
        terminal.draw(|frame| app.render(frame))?;

        // Handle events
        match events.next()? {
            AppEvent::Key(key) => {
                if app.handle_key(key) {
                    break; // App wants to quit
                }
            }
            AppEvent::Tick => {
                app.tick();
            }
            AppEvent::Resize(_, _) => {
                // Terminal will auto-resize on next draw
            }
        }
    }

    Ok(())
}
