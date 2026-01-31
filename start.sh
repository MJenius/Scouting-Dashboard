#!/bin/bash

# Start FastAPI backend in the background
echo "🚀 Starting Backend (FastAPI)..."
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for Backend to be healthy
echo "⏳ Waiting for Backend to initialize..."
max_retries=30
counter=0

while ! curl -s -f http://localhost:8000/health > /dev/null; do
    counter=$((counter+1))
    if [ $counter -gt $max_retries ]; then
        echo "❌ Backend failed to start after ${max_retries} seconds."
        kill $BACKEND_PID
        exit 1
    fi
    sleep 1
    echo "   ...waiting (${counter}/${max_retries})"
done

echo "✅ Backend is healthy!"

# Start Streamlit frontend
echo "🚀 Starting Frontend (Streamlit)..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
FRONTEND_PID=$!

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
