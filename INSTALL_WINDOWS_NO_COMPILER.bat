@echo off
echo ================================================
echo 🚀 INSTALLING RAG CHATBOT - NO C++ COMPILERS NEEDED
echo ================================================

echo.
echo Step 1: Creating virtual environment...
python -m venv venv_simple
call venv_simple\Scripts\activate.bat

echo.
echo Step 2: Installing NO-COMPILER packages...
pip install --upgrade pip
pip install streamlit==1.28.0
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install pymupdf==1.23.8

echo.
echo Step 3: Optional - for better embeddings (but works without):
echo pip install sentence-transformers==2.2.2
echo.

echo ================================================
echo ✅ INSTALLATION COMPLETE!
echo ================================================
echo.
echo 🎯 Your RAG chatbot is ready!
echo 📁 Put PDF files in the same folder as the script
echo 🚀 Run: streamlit run ULTRA_SIMPLE_RAG.py
echo.
echo 📞 No C++ compilers needed!
echo    No llama-cpp-python!
echo    No ChromaDB issues!
echo.
pause