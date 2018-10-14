lint:
	mypy .

test: lint
	py.test --pep8 -m pep8
	py.test