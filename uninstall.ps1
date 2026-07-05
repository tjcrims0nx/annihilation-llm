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

Write-Host "ANNIHILATE environments have been safely uninstalled." -ForegroundColor Green
