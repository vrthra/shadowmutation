export PYTHONPATH := $(shell pwd):$(PYTHONPATH)

all: tmp
	python3 ./ast_mutator.py examples/bank.py tmp/bank_mut.py
	python3 ./tmp/bank_mut.py

do-2.1: tmp
	python3 ./ast_mutator.py examples/bank.py tmp/bank_mut.py
	LOGICAL_PATH=2.1 python3 ./tmp/bank_mut.py

test:
	pytest --log-cli-level=INFO

tmp:
	mkdir tmp

