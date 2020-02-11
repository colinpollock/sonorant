
mypy:
	mypy sonorous/ --config-file=mypy.ini

test:
	python -m pytest tests/
