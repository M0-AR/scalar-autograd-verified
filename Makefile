.PHONY: test experiments verify fetch-data report docker-verify clean

PY ?= python

test:
	$(PY) -m pytest tests/ -q

experiments:
	$(PY) experiments/run_all.py

report:
	$(PY) -c "from micrograd_verified import report; report.main()"

verify: test experiments

fetch-data:
	$(PY) scripts/fetch_data.py

docker-verify:
	docker compose run --rm verify

clean:
	rm -f results/e*.json results/RESULTS.md results/summary.json
	rm -f results/figures/*.png
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
