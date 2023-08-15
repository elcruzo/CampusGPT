#!/bin/bash

echo "🚀 Setting up CampusGPT with LLaMA 2..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

echo "📦 Installing PyTorch for Mac (MPS support)..."
# For Mac M1/M2 with MPS (Metal Performance Shaders) support
pip3 install torch torchvision torchaudio

echo "📦 Installing core dependencies..."
pip install transformers==4.36.0
pip install datasets==2.14.0
pip install accelerate==0.24.1
pip install peft==0.7.1
pip install bitsandbytes==0.41.3
pip install sentencepiece==0.1.99
pip install protobuf==3.20.3

echo "📦 Installing training dependencies..."
pip install trl==0.7.4
pip install wandb==0.16.0
pip install evaluate==0.4.1
pip install rouge-score==0.1.2
pip install scipy

echo "📦 Installing API dependencies..."
pip install fastapi==0.104.1
pip install uvicorn[standard]==0.24.0
pip install pydantic==2.5.0

echo "📦 Installing Jupyter..."
pip install jupyter jupyterlab ipywidgets

echo "📦 Installing additional utilities..."
pip install python-dotenv pyyaml pandas numpy matplotlib seaborn plotly

echo "✅ Setup complete! Now you need to:"
echo "1. Get HuggingFace token: https://huggingface.co/settings/tokens"
echo "2. Request LLaMA 2 access: https://ai.meta.com/resources/models-and-libraries/llama-downloads/"
echo "3. Accept license on HuggingFace: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf"
echo ""
echo "Run: source venv/bin/activate"