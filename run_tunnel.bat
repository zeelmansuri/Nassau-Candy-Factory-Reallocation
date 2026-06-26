@echo off
cd /d "%~dp0"
"C:\Program Files\nodejs\npx.cmd" --yes localtunnel --port 8501 --local-host 127.0.0.1 --subdomain nassau-candy-opt-8501 >> outputs\localtunnel.fixed.log 2>&1
