mypy:
	mypy sonorant/ --config-file=.mypy.ini

test:
	python -m pytest tests/

black:
	black sonorant tests

lint:
	flake8 sonorant tests

	# Skipping B322, which doesn't apply to Python 3.
	bandit -r sonorant --skip B322

runserver:
	python -m sonorant
