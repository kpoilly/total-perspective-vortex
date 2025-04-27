all:
	pip install uv
	python3 -m uv venv
	source .venv/bin/activate
	uv pip install -r requirements.txt 
	mkdir data/raw data/preprocessed
	wget gdrive data/raw