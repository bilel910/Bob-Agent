@echo off
REM Batch script to replace Makefile functionality for Windows Command Prompt
REM Usage: run.bat <command>

setlocal enabledelayedexpansion

REM Check if .env file exists
if not exist ".env" (
    echo .env file is missing. Please create one based on .env.example
    exit /b 1
)

REM Load environment variables from .env file
for /f "tokens=1,2 delims==" %%a in (.env) do (
    if not "%%a"=="" if not "%%a:~0,1%"=="#" (
        set "%%a=%%b"
    )
)


