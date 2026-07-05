//! # Event Handler Module
//!
//! Provides a simple, synchronous event polling abstraction over
//! [`crossterm::event`].  The [`EventHandler`] converts raw terminal events
//! into the application-level [`AppEvent`] enum, collapsing unhandled
//! variants into [`AppEvent::Tick`] so the main loop always has something
//! to process.

use std::time::Duration;

use color_eyre::Result;
use crossterm::event::{self, Event, KeyEvent, KeyEventKind};

/// Application-level events consumed by the main loop.
#[derive(Debug)]
pub enum AppEvent {
    /// A keyboard event was received.
    Key(KeyEvent),

    /// No meaningful event arrived within the tick window – use this to
    /// drive animations, status refreshes, and other periodic work.
    Tick,

    /// The terminal was resized to (`width`, `height`).
    Resize(u16, u16),
}

/// Polls [`crossterm`] for terminal events at a fixed tick rate.
///
/// # Example
///
/// ```rust,no_run
/// use annihilate::events::{EventHandler, AppEvent};
///
/// let events = EventHandler::new(250); // 250 ms tick
///
/// loop {
///     match events.next().unwrap() {
///         AppEvent::Key(key) => { /* handle input */ }
///         AppEvent::Tick      => { /* update state */ }
///         AppEvent::Resize(..) => { /* redraw */ }
///     }
/// }
/// ```
pub struct EventHandler {
    /// Maximum time to block while waiting for the next event.
    tick_rate: Duration,
}

impl EventHandler {
    /// Creates a new handler that polls every `tick_rate_ms` milliseconds.
    pub fn new(tick_rate_ms: u64) -> Self {
        Self {
            tick_rate: Duration::from_millis(tick_rate_ms),
        }
    }

    /// Blocks for at most [`tick_rate`](Self) and returns the next event.
    ///
    /// * If a key press arrives → [`AppEvent::Key`]
    /// * If the terminal is resized → [`AppEvent::Resize`]
    /// * Otherwise (timeout **or** unhandled event) → [`AppEvent::Tick`]
    pub fn next(&self) -> Result<AppEvent> {
        if event::poll(self.tick_rate)? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => Ok(AppEvent::Key(key)),
                Event::Resize(w, h) => Ok(AppEvent::Resize(w, h)),
                _ => Ok(AppEvent::Tick),
            }
        } else {
            Ok(AppEvent::Tick)
        }
    }
}
