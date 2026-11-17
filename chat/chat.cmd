@echo off
rem Gemma 4 E4B chat on this PC. Double-click this file, or run it from a terminal:
rem   chat.cmd [-Device auto^|cuda^|webgpu^|cpu] [-Adapter <n>] [-Context 2048^|4096^|8192] [-ModelDir <folder>]
rem            [-Log <file>] [-SetupOnly]
rem The first run sets everything up next to this file (see chat.ps1); later runs open the chat at once.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0chat.ps1" %*
exit /b %ERRORLEVEL%
