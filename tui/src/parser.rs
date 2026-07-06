//! Output parser for `annihilate` CLI stdout/stderr.
//!
//! Extracts structured data from Rich-formatted terminal output
//! including trial progress, metrics, timing, and status messages.

/// Parsed event from an annihilate output line.
#[derive(Debug, Clone)]
pub enum ParsedEvent {
    /// Model is being loaded
    ModelLoading(String),
    /// Batch size was determined
    BatchSize(usize),
    /// Dataset loading status
    DatasetLoading(String),
    /// Refusal direction calculation started
    CalculatingDirections,
    /// Optimization starting
    OptimizationStarting { n_trials: usize },
    /// A trial completed with metrics
    TrialComplete {
        trial_number: usize,
        total_trials: usize,
        refusals: usize,
        total_prompts: usize,
        kl_divergence: f64,
    },
    /// Best trial so far updated
    BestTrial {
        trial_number: usize,
        refusals: usize,
        kl_divergence: f64,
    },
    /// Optimization finished
    OptimizationComplete,
    /// GPU memory info
    GpuMemory { used_gb: f64, total_gb: f64 },
    /// Elapsed time
    ElapsedTime(String),
    /// ETA
    EstimatedRemaining(String),
    /// Trial was pruned
    TrialPruned { trial_number: usize },
    /// Error message
    Error(String),
    /// Warning message
    Warning(String),
    /// Generic status message
    Status(String),
    /// Interactive prompt detected (questionary)
    InteractivePrompt(String),
    /// Unrecognized line
    Raw(String),
}

/// Strip ANSI escape codes from a string.
pub fn strip_ansi(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut in_escape = false;

    for ch in s.chars() {
        if in_escape {
            if ch.is_ascii_alphabetic() {
                in_escape = false;
            }
        } else if ch == '\x1b' {
            in_escape = true;
        } else {
            result.push(ch);
        }
    }

    result
}

/// Parse a single line of annihilate output into a structured event.
pub fn parse_line(raw: &str) -> ParsedEvent {
    let line = strip_ansi(raw).trim().to_string();

    if line.is_empty() {
        return ParsedEvent::Raw(String::new());
    }

    // Model loading
    if line.contains("Loading model") || line.contains("loading model") {
        return ParsedEvent::ModelLoading(line.clone());
    }

    // Batch size determination
    if line.contains("batch size") || line.contains("Batch size") {
        if let Some(size) = extract_number_after(&line, "batch size") {
            return ParsedEvent::BatchSize(size as usize);
        }
        return ParsedEvent::Status(line);
    }

    // Dataset loading
    if line.contains("Loading") && (line.contains("prompts") || line.contains("dataset")) {
        return ParsedEvent::DatasetLoading(line.clone());
    }

    // Refusal directions
    if line.contains("refusal direction") || line.contains("Refusal direction") {
        return ParsedEvent::CalculatingDirections;
    }

    // Optimization starting
    if line.contains("Running") && line.contains("trial") {
        // "Running trial 5 of 200"
        return ParsedEvent::Status(line.to_string());
    }

    // Trial results - look for refusal counts and KL divergence
    // Patterns: "Refusals: X/Y" or "refusals: X" or "KL divergence: X.XXXX"
    if line.contains("efusal") && (line.contains('/') || line.contains(':')) {
        let refusals = extract_fraction(&line, "efusal");
        let kl = extract_float_after(&line, "KL");
        if let Some((num, denom)) = refusals {
            return ParsedEvent::TrialComplete {
                trial_number: 0, // Will be updated by context
                total_trials: 0,
                refusals: num,
                total_prompts: denom,
                kl_divergence: kl.unwrap_or(0.0),
            };
        }
    }

    // KL divergence standalone
    if line.contains("KL divergence") || line.contains("kl_divergence") {
        if let Some(kl) = extract_float_after(&line, "KL") {
            return ParsedEvent::Status(format!("KL divergence: {:.4}", kl));
        }
    }

    // GPU memory
    if line.contains("GPU") && line.contains("GB") && line.contains("allocated") {
        return ParsedEvent::Status(line);
    }

    // Optimization complete
    if line.contains("Optimization complete")
        || line.contains("optimization complete")
        || line.contains("Pareto")
    {
        return ParsedEvent::OptimizationComplete;
    }

    // Trial pruned
    if line.contains("pruned") || line.contains("Pruned") {
        return ParsedEvent::Status(line);
    }

    // Error detection
    if line.starts_with("Error") || line.starts_with("ERROR") || line.contains("error:") {
        return ParsedEvent::Error(line);
    }

    // Warning detection
    if line.starts_with("Warning") || line.starts_with("WARNING") || line.contains("warning:") {
        return ParsedEvent::Warning(line);
    }

    // Questionary/interactive prompt detection
    if line.contains("?") && (line.contains("Select") || line.contains("Choose") || line.contains("What")) {
        return ParsedEvent::InteractivePrompt(line);
    }

    // Everything else
    ParsedEvent::Raw(line)
}

/// Extract a number appearing after a keyword.
fn extract_number_after(s: &str, keyword: &str) -> Option<f64> {
    if let Some(pos) = s.to_lowercase().find(&keyword.to_lowercase()) {
        let after = &s[pos + keyword.len()..];
        for word in after.split_whitespace() {
            let cleaned: String = word.chars().filter(|c| c.is_ascii_digit() || *c == '.').collect();
            if let Ok(n) = cleaned.parse::<f64>() {
                return Some(n);
            }
        }
    }
    None
}

/// Extract a float appearing after a keyword.
fn extract_float_after(s: &str, keyword: &str) -> Option<f64> {
    extract_number_after(s, keyword)
}

/// Extract a fraction like "5/100" appearing after a keyword.
fn extract_fraction(s: &str, keyword: &str) -> Option<(usize, usize)> {
    if let Some(pos) = s.to_lowercase().find(&keyword.to_lowercase()) {
        let after = &s[pos..];
        for word in after.split_whitespace() {
            if word.contains('/') {
                let parts: Vec<&str> = word.split('/').collect();
                if parts.len() == 2 {
                    let num: String = parts[0].chars().filter(|c| c.is_ascii_digit()).collect();
                    let denom: String = parts[1].chars().filter(|c| c.is_ascii_digit()).collect();
                    if let (Ok(n), Ok(d)) = (num.parse::<usize>(), denom.parse::<usize>()) {
                        return Some((n, d));
                    }
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_ansi() {
        assert_eq!(strip_ansi("\x1b[32mHello\x1b[0m"), "Hello");
        assert_eq!(strip_ansi("No escapes"), "No escapes");
    }

    #[test]
    fn test_parse_trial() {
        match parse_line("Running trial 5 of 200") {
            ParsedEvent::TrialComplete { trial_number: 5, total_trials: 200, .. } => {},
            other => panic!("Expected TrialComplete, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_refusals() {
        match parse_line("Refusals: 3/100, KL divergence: 0.0312") {
            ParsedEvent::TrialComplete { refusals: 3, total_prompts: 100, .. } => {},
            other => panic!("Expected TrialComplete, got {:?}", other),
        }
    }
}
