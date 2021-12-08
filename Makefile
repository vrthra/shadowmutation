PHONY: bank test do-2.1

export PYTHONPATH := $(shell pwd):$(PYTHONPATH)

bank: tmp
	python3 ./ast_mutator.py --ignore "^test_" examples/bank.py tmp/bank_mut.py
	python3 ./tmp/bank_mut.py

do-2.1: tmp
	python3 ./ast_mutator.py examples/bank.py tmp/bank_mut.py
	LOGICAL_PATH=2.1 python3 ./tmp/bank_mut.py

test:
	pytest --log-cli-level=DEBUG --log-format="%(levelname)s %(process)d %(message)s"



tmp:
	mkdir tmp

