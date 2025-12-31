# RAG Orchestrator - Start All Services
# Run this in PowerShell: .\start_services.ps1

Write-Host "Starting RAG Orchestrator Services..." -ForegroundColor Cyan
Write-Host ""

if (-not $env:VIRTUAL_ENV) {
    Write-Host "Virtual environment not activated!" -ForegroundColor Yellow
    Write-Host "Run: .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host ""
}

function Test-Port {
    param($Port)
    $connection = Test-NetConnection -ComputerName localhost -Port $Port -InformationLevel Quiet
    return $connection
}

Write-Host "Checking Qdrant (port 6333)..." -ForegroundColor Yellow
if (Test-Port -Port 6333) {
    Write-Host "Qdrant is already running" -ForegroundColor Green
} else {
    Write-Host "Qdrant is not running" -ForegroundColor Red
    Write-Host "Start Qdrant with: docker run -p 6333:6333 qdrant/qdrant" -ForegroundColor Yellow
    $startQdrant = Read-Host "Do you want to start Qdrant now? (y/n)"
    if ($startQdrant -eq "y") {
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "docker run -p 6333:6333 qdrant/qdrant"
        Write-Host "Waiting for Qdrant to start..." -ForegroundColor Yellow
        Start-Sleep -Seconds 5
    }
}

Write-Host ""

Write-Host "Starting FastAPI + Inngest backend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; Write-Host 'FastAPI + Inngest Backend' -ForegroundColor Cyan; uv run uvicorn main:app --reload"
Start-Sleep -Seconds 2

Write-Host "Starting Inngest Dev Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; Write-Host 'Inngest Dev Server' -ForegroundColor Magenta; npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery"
Start-Sleep -Seconds 3

Write-Host "Starting Streamlit Frontend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; Write-Host 'Streamlit Frontend' -ForegroundColor Green; uv run streamlit run streamlit_app.py"
Start-Sleep -Seconds 2

Write-Host ""
Write-Host "All services started!" -ForegroundColor Green
Write-Host ""
Write-Host "Access Points:" -ForegroundColor Cyan
Write-Host "  Streamlit:  http://localhost:8501" -ForegroundColor White
Write-Host "  Inngest:    http://localhost:8288" -ForegroundColor White
Write-Host "  FastAPI:    http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "  Qdrant:     http://localhost:6333/dashboard" -ForegroundColor White
Write-Host ""
Write-Host "Tip: Keep all terminal windows open while using the app" -ForegroundColor Yellow
Write-Host "Press Ctrl+C in each window to stop services" -ForegroundColor Yellow
Write-Host ""