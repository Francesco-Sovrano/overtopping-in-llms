python3.12 -m venv .env

. .env/bin/activate

pip install -U pip
pip install -U setuptools wheel
pip install -U -r requirements.txt

ollama pull gemma3:27b
ollama pull qwen3:4b
ollama pull qwen3:14b