@echo off
echo Setting up Offline RAG Chatbot...
echo.

REM Create virtual environment
python -m venv venv
call venv\Scripts\activate.bat

REM Install requirements
pip install -r requirements.txt

REM Create directories
mkdir data\pdfs 2>nul
mkdir data\database 2>nul
mkdir data\models 2>nul

echo.
echo ✅ Setup complete!
echo.
echo 📚 Place your PDF files in the 'data\pdfs' folder
echo 🚀 Run the app by opening 'simple_app.py' in VS Code
echo.
echo Press any key to open VS Code...
pause >nul
code .
