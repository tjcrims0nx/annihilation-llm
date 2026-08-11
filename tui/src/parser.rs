//! Output parser for `annihilate` CLI stdout/stderr.
//!
//! Extracts structured data from Rich-formatted terminal output
//! including trial progress, metrics, timing, and status messages.

/// Parsed event from an annihilate output line.
#[derive(Debug, Clone)]
pub enum ParsedEvent {
    /// Model is being loaded
    ModelLoading(String),
    /// Architecture detected from the model's config, before weights load
    ModelFormat {
        architecture: String,
        multimodal: bool,
        remote_code: bool,
    },
    /// Model ships already quantized, with the method it declares
    Quantization(String),
    /// Batch size was determined
    BatchSize(usize),
    /// Dataset loading status
    DatasetLoading(String),
    /// Refusal direction calculation started
    CalculatingDirections,
    /// Optimization starting
    OptimizationStarting {
        n_trials: usize,
    },
    /// Trial started
    TrialStarting {
        trial_number: usize,
        total_trials: usize,
    },
    /// A trial completed with metrics
    TrialComplete {
        trial_number: usize,
        total_trials: usize,
        refusals: usize,
        total_prompts: usize,
    },
    KLDivergence(f64),
    /// Best trial so far updated
    BestTrial {
        trial_number: usize,
        refusals: usize,
        kl_divergence: f64,
    },
    /// Optimization finished
    OptimizationComplete,
    /// GPU memory info
    GpuMemory {
        used_gb: f64,
        total_gb: f64,
    },
    /// Elapsed time
    ElapsedTime(String),
    /// ETA
    EstimatedRemaining(String),
    /// Trial was pruned
    TrialPruned {
        trial_number: usize,
    },
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

    // JSON protocol lines from chat/benchmark/GGUF scripts must stay Raw.
    // Otherwise phrases like "Loading model" inside JSON get misclassified
    // and the TUI never sees ready/status/token/result events.
    if line.starts_with('{') && line.ends_with('}') {
        return ParsedEvent::Raw(line);
    }

    // Model loading
    if line.contains("Loading model") || line.contains("loading model") {
        return ParsedEvent::ModelLoading(line.clone());
    }

    // Format detection, emitted before the weights download. Matched ahead of the
    // generic rules below because "* Detected 1 CUDA device(s)" would otherwise
    // collide: require the architecture form specifically.
    if let Some(rest) = line.strip_prefix("* Detected ")
        && !rest.contains("CUDA")
    {
        let mut parts = rest.split(',').map(str::trim);
        let architecture = parts.next().unwrap_or_default().to_string();
        let flags: Vec<&str> = parts.collect();
        return ParsedEvent::ModelFormat {
            architecture,
            multimodal: flags.contains(&"multimodal"),
            remote_code: flags.contains(&"custom code"),
        };
    }

    if let Some(method) = line.strip_prefix("* Pre-quantized model: ") {
        return ParsedEvent::Quantization(method.trim().to_string());
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

    // Trial starting
    if line.contains("Running") && line.contains("trial") && line.contains("of") {
        // "Running trial 5 of 200..."
        let parts: Vec<&str> = line.split_whitespace().collect();
        let mut trial_num = 0;
        let mut total_trials = 0;

        for (i, part) in parts.iter().enumerate() {
            if part.contains("trial")
                && i + 1 < parts.len()
                && let Ok(n) = parts[i + 1].replace(',', "").parse()
            {
                trial_num = n;
            }
            if part.contains("of") && i + 1 < parts.len() {
                let stripped: String = parts[i + 1]
                    .chars()
                    .filter(|c| c.is_ascii_digit())
                    .collect();
                if let Ok(n) = stripped.parse() {
                    total_trials = n;
                }
            }
        }

        if trial_num > 0 && total_trials > 0 {
            return ParsedEvent::TrialStarting {
                trial_number: trial_num,
                total_trials,
            };
        }
        return ParsedEvent::Status(line.to_string());
    }

    // Trial results - look for refusal counts and KL divergence
    // Patterns: "Refusals: X/Y" or "refusals: X" or "KL divergence: X.XXXX"
    if line.contains("efusal") && (line.contains('/') || line.contains(':')) {
        let refusals = extract_fraction(&line, "efusal");
        if let Some((num, denom)) = refusals {
            return ParsedEvent::TrialComplete {
                trial_number: 0, // Will be updated by context
                total_trials: 0,
                refusals: num,
                total_prompts: denom,
            };
        }
    }

    // KL divergence standalone
    if (line.contains("KL divergence") || line.contains("kl_divergence"))
        && let Some(kl) = extract_float_after(&line, "KL")
    {
        return ParsedEvent::KLDivergence(kl);
    }

    // GPU memory
    if line.contains("GPU") && line.contains("GB") && line.contains("allocated") {
        return ParsedEvent::Status(line);
    }

    // Optimization complete
    if line.contains("Optimization complete")
        || line.contains("optimization complete")
        || line.contains("Optimization finished")
        || line.contains("Optimization interrupted by user")
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
    if line.contains("?")
        && (line.contains("Select") || line.contains("Choose") || line.contains("What"))
    {
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
            let cleaned: String = word
                .chars()
                .filter(|c| c.is_ascii_digit() || *c == '.')
                .collect();
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
            ParsedEvent::TrialStarting {
                trial_number: 5,
                total_trials: 200,
            } => {}
            other => panic!("Expected TrialStarting, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_refusals() {
        match parse_line("Refusals: 3/100, KL divergence: 0.0312") {
            ParsedEvent::TrialComplete {
                refusals: 3,
                total_prompts: 100,
                ..
            } => {}
            other => panic!("Expected TrialComplete, got {:?}", other),
        }
    }

    #[test]
    fn test_json_protocol_stays_raw() {
        let status = r#"{"type": "status", "content": "Loading model..."}"#;
        match parse_line(status) {
            ParsedEvent::Raw(line) => assert_eq!(line, status),
            other => panic!("Expected Raw JSON status, got {:?}", other),
        }

        let ready = r#"{"type": "ready"}"#;
        match parse_line(ready) {
            ParsedEvent::Raw(line) => assert_eq!(line, ready),
            other => panic!("Expected Raw JSON ready, got {:?}", other),
        }

        let refusal_status =
            r#"{"type": "status", "content": "Calculating refusal directions..."}"#;
        match parse_line(refusal_status) {
            ParsedEvent::Raw(line) => assert_eq!(line, refusal_status),
            other => panic!("Expected Raw JSON refusal status, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_model_format() {
        match parse_line("* Detected LlamaForCausalLM") {
            ParsedEvent::ModelFormat {
                architecture,
                multimodal: false,
                remote_code: false,
            } => assert_eq!(architecture, "LlamaForCausalLM"),
            other => panic!("Expected ModelFormat, got {:?}", other),
        }

        match parse_line("* Detected CustomVLM, multimodal, custom code") {
            ParsedEvent::ModelFormat {
                architecture,
                multimodal: true,
                remote_code: true,
            } => assert_eq!(architecture, "CustomVLM"),
            other => panic!("Expected multimodal ModelFormat, got {:?}", other),
        }
    }

    #[test]
    fn test_cuda_detection_is_not_a_model_format() {
        // "* Detected N CUDA device(s)" shares the prefix but is not an
        // architecture line; misreading it would show "1 CUDA device(s)" as
        // the model's architecture on the dashboard.
        assert!(
            !matches!(
                parse_line("* Detected 1 CUDA device(s) (4.00 GB total VRAM)"),
                ParsedEvent::ModelFormat { .. }
            ),
            "CUDA device line must not parse as a model format"
        );
    }

    #[test]
    fn test_parse_quantization() {
        match parse_line("* Pre-quantized model: compressed-tensors") {
            ParsedEvent::Quantization(method) => assert_eq!(method, "compressed-tensors"),
            other => panic!("Expected Quantization, got {:?}", other),
        }
    }
}
