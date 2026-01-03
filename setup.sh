#!/bin/bash
set -e

echo "Setting up Offline RAG Chatbot..."
echo ""

# Create virtual environment
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Create directories
mkdir -p data/pdfs
mkdir -p data/database
mkdir -p data/models

echo ""
echo "✅ Setup complete!"
echo ""
echo "📚 Place your PDF files in the 'data/pdfs' folder"
echo "🚀 Run the app by opening 'simple_app.py' in VS Code or run 'streamlit run ui/app.py'"
echo ""
echo "Opening VS Code..."
if command -v code >/dev/null 2>&1; then
  code .
else
  echo "(Install 'code' CLI or open the folder in VS Code manually.)"
fi
