@echo off
echo Starting Backend...
start "Backend Service" cmd /c "cd backend && mvn spring-boot:run"

echo Starting Frontend...
start "Frontend Service" cmd /c "cd frontend && npm run serve"

echo ---------------------------------------------------
echo System started! Checks the new command windows for logs.
echo ---------------------------------------------------
pause
