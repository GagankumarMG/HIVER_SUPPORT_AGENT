.PHONY: install fast full evaluate test lint clean demo synthetic-data

install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

synthetic-data:
	python scripts/00b_make_synthetic_dataset.py

fast:
	python scripts/run_all.py --fast

full:
	python scripts/run_all.py --full

evaluate:
	python scripts/08_evaluate.py
	python scripts/09_run_llm_judge.py
	python scripts/09b_human_agreement.py
	python scripts/10_analyze_failures.py
	python scripts/12_leakage_check.py
	python scripts/11_generate_report.py

test:
	pytest -q

demo:
	streamlit run app/streamlit_app.py

clean:
	rm -rf data/interim/* data/processed/* models/* outputs/figures/* outputs/metrics/* outputs/examples/*
