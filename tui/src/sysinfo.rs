//! Real system hardware detection.
//!
//! Queries GPU info via `nvidia-smi`, CPU brand via the native platform
//! command (Win32_Processor on Windows, `/proc/cpuinfo` on Linux,
//! `sysctl -n machdep.cpu.brand_string` on macOS), and RAM usage via
//! platform commands (PowerShell on Windows, `free` on Linux, `vm_stat` +
//! `hw.memsize` on macOS).

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

        if let Ok(output) = output
            && output.status.success()
        {
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

        #[cfg(target_os = "linux")]
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

        #[cfg(target_os = "macos")]
        {
            // macOS has no /proc; `machdep.cpu.brand_string` is the native
            // equivalent ("Apple M4" on Apple Silicon, the Intel brand string
            // on older Macs).
            if let Ok(output) = Command::new("sysctl")
                .args(["-n", "machdep.cpu.brand_string"])
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

        #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
        {
            // Unknown Unix: leave the nvidia-smi result if one was found,
            // otherwise stop at a concrete label instead of "Detecting...".
            self.gpu_name = "CPU (Unknown)".to_string();
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

        #[cfg(target_os = "linux")]
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

        #[cfg(target_os = "macos")]
        {
            // macOS has no `free`. Total comes from `sysctl -n hw.memsize`
            // (bytes); used is what Activity Monitor counts as used memory:
            // active + wired + compressed pages from `vm_stat`.
            if let Ok(total_out) = Command::new("sysctl").args(["-n", "hw.memsize"]).output()
                && total_out.status.success()
            {
                let total_bytes: f64 = String::from_utf8_lossy(&total_out.stdout)
                    .trim()
                    .parse()
                    .unwrap_or(0.0);
                if total_bytes > 0.0 {
                    self.ram_total_mb = total_bytes / (1024.0 * 1024.0);
                }
            }

            if let Ok(vm_out) = Command::new("vm_stat").output()
                && vm_out.status.success()
            {
                let text = String::from_utf8_lossy(&vm_out.stdout);
                let mut page_size: f64 = 4096.0;
                let mut active: f64 = 0.0;
                let mut wired: f64 = 0.0;
                let mut compressed: f64 = 0.0;

                for line in text.lines() {
                    let digits: String = line.matches(char::is_numeric).collect();

                    if line.contains("page size") {
                        // "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
                        page_size = digits.parse().unwrap_or(4096.0);
                    } else if let Ok(pages) = digits.parse::<f64>() {
                        if line.starts_with("Pages active:") {
                            active = pages;
                        } else if line.starts_with("Pages wired down:") {
                            wired = pages;
                        } else if line.starts_with("Pages occupied by compressor:") {
                            compressed = pages;
                        }
                    }
                }

                self.ram_used_mb = (active + wired + compressed) * page_size / (1024.0 * 1024.0);
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
