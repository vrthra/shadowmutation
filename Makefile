.PHONY: all bank test do-2.1

export PYTHONPATH := $(shell pwd):$(PYTHONPATH)

EXAMPLES := $(wildcard examples/*.py)
SUBJECTS := $(notdir $(EXAMPLES:.py=))

all: $(SUBJECTS)

$(SUBJECTS): tmp
	@echo $@
	rm -r tmp/$@ || true
	python3 ./ast_mutator.py --ignore "^test_" examples/$@.py tmp/$@
	python execute_versions.py tmp/$@


dev:
	EXECUTION_MODE=shadow python3 tmp/prime/shadow_execution.py

test:
	pytest --log-cli-level=DEBUG --log-format="%(levelname)s %(process)d %(message)s"


tmp:
	mkdir tmp

