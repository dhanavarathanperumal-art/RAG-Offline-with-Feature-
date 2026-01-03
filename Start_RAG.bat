@echo off
title RAG Chatbot Launcher
echo ========================================
echo   STARTING OFFLINE RAG CHATBOT
echo ========================================
echo.

cd /d "C:\Users\USER\Documents\project dhana\offline Rag bot"

echo Activating virtual environment...
call venv_simple\Scripts\activate

echo.
echo Starting RAG Chatbot...
echo.
echo ^(1^) Streamlit will open in your browser
echo ^(2^) If browser doesn't open automatically:
echo     Go to: http://localhost:8501
echo ^(3^) Press CTRL+C in this window to stop
echo.
echo ========================================
echo.

streamlit run ULTRA_SIMPLE_RAG.py

pause