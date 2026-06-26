@echo off
cd /d "%~dp0"
"D:\python\python.exe" -m streamlit run app.py --server.port 8501 --server.headless true >> outputs\streamlit.detached.log 2>&1
