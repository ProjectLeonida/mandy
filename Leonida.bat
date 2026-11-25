@echo off
rem Opens the Leonida builder window (Build.bat is the console version).
cd /d "%~dp0"
where pyw >nul 2>nul && (start "" pyw -3 -I "%~dp0leonida_app.py" & exit /b 0)
where pythonw >nul 2>nul && (start "" pythonw -I "%~dp0leonida_app.py" & exit /b 0)
echo Python 3.12 or newer with tkinter is needed. Run Build.bat to install it, then try again.
pause
