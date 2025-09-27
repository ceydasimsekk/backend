@echo off
echo Starting Emotify application...

REM Start the Python API server in a new window
start cmd /k "python api.py"

REM Wait a moment for the API to start
timeout /t 3

REM Start the Next.js frontend
npm run dev
