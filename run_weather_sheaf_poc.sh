#!/bin/bash

echo -e "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "🚀 Weather Forecasting System - Startup Sequence"
echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
echo -e "📝 First-Time Setup:"
echo -e "   • First run will download AFNONet model (~900MB)"
echo -e "   • Wait for 'Server listening on port 5001...'"
echo -e "   • Then copy the localhost URL from terminal to browser\n"

lsof -ti:5001 | xargs kill -9 2>/dev/null

echo "Starting Inference Server..."
python inference_server.py &
SERVER_PID=$!

while ! nc -z localhost 5001 2>/dev/null; do
    sleep 1
done

echo -e "\n✅ Server ready! Starting client interface..."
streamlit run client_app.py &
CLIENT_PID=$!

cleanup() {
    echo -e "\n💫 Shutting down applications..."
    kill -9 $SERVER_PID 2>/dev/null
    kill -9 $CLIENT_PID 2>/dev/null
    pkill -f "python inference_server.py" 2>/dev/null
    pkill -f "streamlit run client_app.py" 2>/dev/null
    lsof -ti:5001 | xargs kill -9 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

echo -e "\n⚡ System running. Press Ctrl+C to stop."
wait