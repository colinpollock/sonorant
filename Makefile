mypy:
	mypy sonorous/ --config-file=.mypy.ini

test:
	python -m pytest tests/

black:
	black sonorous tests

lint:
	flake8 sonorous tests

	# Skipping B322, which doesn't apply to Python 3.
	bandit -r sonorous --skip B322

runserver:
	python -m sonorous
