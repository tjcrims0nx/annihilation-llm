$InstallDir = "$env:USERPROFILE\annihilation-llm"
Write-Host "Installing ANNIHILATE to $InstallDir..." -ForegroundColor Cyan

if (Test-Path $InstallDir) {
    Write-Host "Directory already exists. Updating repository..."
    Set-Location $InstallDir
    git pull
} else {
    git clone https://github.com/tjcrims0nx/annihilation-llm.git $InstallDir
    Set-Location $InstallDir
}

Write-Host "Fetching latest release binary..." -ForegroundColor Cyan
$ReleaseUrl = "https://api.github.com/repos/tjcrims0nx/annihilation-llm/releases/latest"
try {
    $ReleaseData = Invoke-RestMethod -Uri $ReleaseUrl
    $WindowsAsset = $ReleaseData.assets | Where-Object { $_.name -eq "annihilate-windows.exe" }
    
    if ($WindowsAsset) {
        if (-not (Test-Path "tui\target\release")) {
            New-Item -ItemType Directory -Force -Path "tui\target\release" | Out-Null
        }
        Invoke-WebRequest -Uri $WindowsAsset.browser_download_url -OutFile "tui\target\release\annihilate.exe"
        Write-Host "Binary downloaded successfully." -ForegroundColor Green
    } else {
        Write-Host "Could not find Windows binary in the latest release." -ForegroundColor Yellow
    }
} catch {
    Write-Host "No GitHub releases found or API limit reached. (If this is the first setup, a Release needs to be published first on GitHub)." -ForegroundColor Yellow
}

$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\ANNIHILATE.lnk")
$Shortcut.TargetPath = "$InstallDir\start.bat"
$Shortcut.WorkingDirectory = $InstallDir
$Shortcut.IconLocation = "cmd.exe"
$Shortcut.Save()

Write-Host "Installation complete! An ANNIHILATE shortcut has been added to your Desktop." -ForegroundColor Green
Write-Host "Double click the shortcut to begin!" -ForegroundColor Green
