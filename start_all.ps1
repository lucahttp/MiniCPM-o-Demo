# MiniCPM-o Full Duplex Launcher for Windows
$ErrorActionPreference = "Stop"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  MiniCPM-o 4.5 Full-Duplex Speech & Vision Launcher" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# 1. Start Worker in background
Write-Host "[1/2] Iniciando Worker en puerto 22400 (C++ / llama-omni-server)..." -ForegroundColor Yellow
$WorkerJob = Start-Job -ScriptBlock {
    param($dir)
    Set-Location $dir
    python worker.py --port 22400
} -ArgumentList $ScriptDir

Write-Host "Esperando a que el Worker cargue el modelo en memoria..." -ForegroundColor Yellow
$Ready = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 2
    try {
        $res = Invoke-RestMethod -Uri "http://localhost:22400/health" -Method Get -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($res.model_loaded -eq $true) {
            $Ready = $true
            break
        }
    } catch {}
    Write-Host -NoNewline "."
}
Write-Host ""

if (-not $Ready) {
    Write-Host "El worker tardo demasiado en responder. Revisa los logs." -ForegroundColor Red
    Receive-Job -Job $WorkerJob
    exit 1
}

Write-Host "[Worker OK] Modelo cargado y listo en GPU/C++." -ForegroundColor Green

# 2. Start Gateway
Write-Host "[2/2] Iniciando Gateway en https://localhost:8006 ..." -ForegroundColor Yellow
Write-Host "Abre tu navegador en: https://localhost:8006" -ForegroundColor Cyan
Write-Host "(Presiona CTRL+C para detener todo)" -ForegroundColor DarkGray

try {
    python gateway.py
} finally {
    Stop-Job -Job $WorkerJob -ErrorAction SilentlyContinue
    Remove-Job -Job $WorkerJob -ErrorAction SilentlyContinue
}
