@echo off

rem Install Python packages
pip install -r requirements.txt

rem Clone the Git repository
git clone https://huggingface.co/nlpconnect/vit-gpt2-image-captioning