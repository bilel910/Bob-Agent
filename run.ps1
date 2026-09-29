# PowerShell script to replace Makefile functionality for Windows
# Usage: .\run.ps1 <command>

param(
    [Parameter(Mandatory=$true)]
    [string]$Command
)

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Error ".env file is missing. Please create one based on .env.example"
    exit 1
}

# Load environment variables from .env file
Get-Content ".env" | ForEach-Object {
    if ($_ -match "^([^#][^=]+)=(.*)$") {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
}

$CHECK_DIRS = "."


