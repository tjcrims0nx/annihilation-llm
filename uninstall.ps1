$InstallDir = "$env:USERPROFILE\annihilation-llm"

Write-Host "Uninstalling ANNIHILATE..." -ForegroundColor Cyan

# Remove the virtual environments
Write-Host "Removing Python environments..."
$Envs = @("annihilation-env", ".venv", "venv", "env")
foreach ($Env in $Envs) {
    $EnvPath = "$InstallDir\$Env"
    if (Test-Path $EnvPath) {
        Write-Host "Deleting $EnvPath..."
        Remove-Item -Recurse -Force $EnvPath
    }
}

# Remove desktop shortcut
$ShortcutPath = "$env:USERPROFILE\Desktop\ANNIHILATE.lnk"
if (Test-Path $ShortcutPath) {
    Write-Host "Removing Desktop shortcut..."
    Remove-Item $ShortcutPath
}

Write-Host "Do you also want to clear downloaded HuggingFace Models? (This may free up gigabytes of space, but models will need to be re-downloaded if you reinstall)" -ForegroundColor Yellow
$response = Read-Host "[y/N]"

if ($response -match "^[yY]$") {
    $CachePath = "$env:USERPROFILE\.cache\huggingface\hub"
    if (Test-Path $CachePath) {
        Write-Host "Clearing HuggingFace model cache..."
        Remove-Item -Recurse -Force $CachePath
        Write-Host "Model cache cleared." -ForegroundColor Green
    } else {
        Write-Host "No HuggingFace cache found."
    }
}

Write-Host "ANNIHILATE has been completely uninstalled." -ForegroundColor Green
