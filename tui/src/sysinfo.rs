//! Real system hardware detection.
//!
//! Queries GPU info via `nvidia-smi` and RAM usage via Windows commands.

use std::process::Command;

/// Detected system hardware information.
#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub gpu_name: String,
    pub vram_total_mb: f64,
    pub vram_used_mb: f64,
    pub ram_total_mb: f64,
    pub ram_used_mb: f64,
}

impl Default for SystemInfo {
    fn default() -> Self {
        Self {
            gpu_name: "Detecting...".to_string(),
            vram_total_mb: 0.0,
            vram_used_mb: 0.0,
            ram_total_mb: 0.0,
            ram_used_mb: 0.0,
        }
    }
}

impl SystemInfo {
    /// Detect all system info (GPU + RAM). Call once at startup.
    pub fn detect() -> Self {
        let mut info = Self::default();
        info.refresh_gpu();
        info.refresh_ram();
        info
    }

    pub fn refresh_gpu(&mut self) {
        let output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output();

        if let Ok(output) = output {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                // Output format: "NVIDIA GeForce RTX 4090, 1234, 24564"
                let line = stdout.trim();
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 3 {
                    self.gpu_name = parts[0].to_string();
                    self.vram_used_mb = parts[1].parse().unwrap_or(0.0);
                    self.vram_total_mb = parts[2].parse().unwrap_or(0.0);
                    return;
                }
            }
        }

        // Fallback to CPU detection
        #[cfg(target_os = "windows")]
        {
            if let Ok(output) = Command::new("powershell")
                .args([
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor).Name",
                ])
                .output()
            {
                if output.status.success() {
                    let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    self.gpu_name = if name.is_empty() {
                        "CPU (Unknown)".to_string()
                    } else {
                        format!("CPU: {}", name)
                    };
                } else {
                    self.gpu_name = "CPU (Unknown)".to_string();
                }
            }
        }

        #[cfg(not(target_os = "windows"))]
        {
            if let Ok(output) = Command::new("sh").args(["-c", "cat /proc/cpuinfo | grep -i 'model name' | head -n 1 | awk -F: '{print $2}' | xargs"]).output() {
                if output.status.success() {
                    let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    self.gpu_name = if name.is_empty() { "CPU (Unknown)".to_string() } else { format!("CPU: {}", name) };
                } else {
                    self.gpu_name = "CPU (Unknown)".to_string();
                }
            }
        }
    }

    pub fn refresh_ram(&mut self) {
        #[cfg(target_os = "windows")]
        {
            let output = Command::new("powershell")
                .args([
                    "-NoProfile",
                    "-Command",
                    "[math]::Round((Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize/1024,0).ToString() + ',' + [math]::Round(((Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize - (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory)/1024,0).ToString()",
                ])
                .output();

            if let Ok(output) = output
                && output.status.success()
            {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let parts: Vec<&str> = stdout.trim().split(',').collect();
                if parts.len() >= 2 {
                    self.ram_total_mb = parts[0].parse().unwrap_or(0.0);
                    self.ram_used_mb = parts[1].parse().unwrap_or(0.0);
                }
            }
        }

        #[cfg(not(target_os = "windows"))]
        {
            let output = Command::new("sh")
                .args(["-c", "free -m | awk '/^Mem:/ {print $2 \",\" $3}'"])
                .output();

            if let Ok(output) = output
                && output.status.success()
            {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let parts: Vec<&str> = stdout.trim().split(',').collect();
                if parts.len() >= 2 {
                    self.ram_total_mb = parts[0].parse().unwrap_or(0.0);
                    self.ram_used_mb = parts[1].parse().unwrap_or(0.0);
                }
            }
        }
    }

    /// Quick GPU refresh using nvidia-smi (just VRAM usage, faster).
    pub fn refresh_vram_quick(&mut self) {
        let output = Command::new("nvidia-smi")
            .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
            .output();

        if let Ok(output) = output
            && output.status.success()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            self.vram_used_mb = stdout.trim().parse().unwrap_or(self.vram_used_mb);
        }
    }

    /// Get VRAM in GB for display.
    pub fn vram_used_gb(&self) -> f64 {
        self.vram_used_mb / 1024.0
    }

    pub fn vram_total_gb(&self) -> f64 {
        self.vram_total_mb / 1024.0
    }

    /// Get RAM in GB for display.
    pub fn ram_used_gb(&self) -> f64 {
        self.ram_used_mb / 1024.0
    }

    pub fn ram_total_gb(&self) -> f64 {
        self.ram_total_mb / 1024.0
    }
}
