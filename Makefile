install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

lint:
	flake8 src/ tests/ --max-line-length=100

test:
	pytest -v

train:
	python -m src.train

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete