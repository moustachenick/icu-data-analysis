init:
	pip install -r requirements.txt

test:
	nosetests tests

lint:
	ruff check .

lint-fix:
	ruff check . --fix