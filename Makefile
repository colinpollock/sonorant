
mypy:
	mypy sonorous/ --config-file=mypy.ini

test:
	python -m pytest tests/

lint:
	- pylint --rcfile=pylintrc sonorous/
	bandit -r sonorous/
