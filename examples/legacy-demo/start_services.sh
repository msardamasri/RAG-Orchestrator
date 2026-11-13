#!/bin/bash
# RAG Orchestrator - Start All Services
# Run this: chmod +x start_services.sh && ./start_services.sh

echo "Starting RAG Orchestrator Services..."
echo ""

if [ -z "$VIRTUAL_ENV" ]; then
    echo "Virtual environment not activated!"
    echo "Run: source .venv/bin/activate"
    echo ""
fi

echo "Checking Qdrant (port 6333)..."
if nc -z localhost 6333 2>/dev/null; then
    echo "Qdrant is already running"
else
    echo "Qdrant is not running"
    echo "Start Qdrant with: docker run -p 6333:6333 qdrant/qdrant"
    read -p "Do you want to start Qdrant now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        gnome-terminal -- bash -c "docker run -p 6333:6333 qdrant/qdrant; exec bash" 2>/dev/null || \
        osascript -e 'tell app "Terminal" to do script "docker run -p 6333:6333 qdrant/qdrant"' 2>/dev/null || \
        xterm -e "docker run -p 6333:6333 qdrant/qdrant" 2>/dev/null &
        echo "Waiting for Qdrant to start..."
        sleep 5
    fi
fi

echo ""

echo "Starting FastAPI + Inngest backend..."
gnome-terminal -- bash -c "echo 'FastAPI + Inngest Backend'; uv run uvicorn main:app --reload; exec bash" 2>/dev/null || \
osascript -e "tell app \"Terminal\" to do script \"cd $(pwd) && echo 'FastAPI + Inngest Backend' && uv run uvicorn main:app --reload\"" 2>/dev/null || \
xterm -e "uv run uvicorn main:app --reload" 2>/dev/null &
sleep 2

echo "Starting Inngest Dev Server..."
gnome-terminal -- bash -c "echo 'Inngest Dev Server'; npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery; exec bash" 2>/dev/null || \
osascript -e "tell app \"Terminal\" to do script \"cd $(pwd) && echo 'Inngest Dev Server' && npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery\"" 2>/dev/null || \
xterm -e "npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery" 2>/dev/null &
sleep 3

echo "Starting Streamlit Frontend..."
gnome-terminal -- bash -c "echo 'Streamlit Frontend'; uv run streamlit run streamlit_app.py; exec bash" 2>/dev/null || \
osascript -e "tell app \"Terminal\" to do script \"cd $(pwd) && echo 'Streamlit Frontend' && uv run streamlit run streamlit_app.py\"" 2>/dev/null || \
xterm -e "uv run streamlit run streamlit_app.py" 2>/dev/null &
sleep 2

echo ""
echo "All services started!"
echo ""
echo "Access Points:"
echo "  Streamlit:  http://localhost:8501"
echo "  Inngest:    http://localhost:8288"
echo "  FastAPI:    http://127.0.0.1:8000/docs"
echo "  Qdrant:     http://localhost:6333/dashboard"
echo ""
echo "Tip: Keep all terminal windows open while using the app"
echo "Press Ctrl+C in each window to stop services"
echo ""