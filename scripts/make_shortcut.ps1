# Creates a "Leonida" shortcut (with the project icon) on the Desktop that runs Build.bat.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\make_shortcut.ps1
$root = Split-Path -Parent $PSScriptRoot
$desktop = [Environment]::GetFolderPath('Desktop')
$lnk = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $desktop 'Leonida.lnk'))
$lnk.TargetPath = Join-Path $root 'Build.bat'
$lnk.WorkingDirectory = $root
$lnk.IconLocation = (Join-Path $root 'assets\logo.ico') + ',0'
$lnk.Description = 'Leonida - Game Builder'
$lnk.Save()
Write-Host "Created $($lnk.FullName)"
