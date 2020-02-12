mypy:
	mypy sonorous/ --config-file=.mypy.ini

test:
	python -m pytest tests/

black:
	black sonorous tests

lint:
	flake8 sonorous tests
	bandit -r sonorous
