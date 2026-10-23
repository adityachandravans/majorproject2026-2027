.PHONY: install train train-smoke evaluate backend frontend test demo clean

install:
	pip install -r requirements.txt
	cd frontend && npm install && npm run build

train:
	python marl/train.py

train-smoke:
	python marl/train.py --smoke

evaluate:
	python marl/evaluate.py --episodes 100

backend:
	python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

frontend:
	cd frontend && npm run dev

test:
	python -m pytest -o pythonpath=.
	cd frontend && npm test

demo:
	python scripts/demo.py

clean:
	rm -rf checkpoints/*.pt results/*.csv results/*.png results/*.md
