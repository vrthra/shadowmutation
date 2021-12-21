.PHONY: all bank test do-2.1

export PYTHONPATH := $(shell pwd):$(PYTHONPATH)

EXAMPLES := $(wildcard examples/*.py)
SUBJECTS := $(notdir $(EXAMPLES:.py=))

all: $(SUBJECTS)

clean:
	rm -rf tmp

$(SUBJECTS): | tmp
	@echo $@
	python3 ./ast_mutator.py --ignore "^test_" examples/$@.py tmp/$@
	python execute_versions.py tmp/$@


dev:
	# TRACE=1 python3 tmp/prime/traditional_1.py
	EXECUTION_MODE=split GATHER_ATEXIT=1 TRACE=1 python3 tmp/prime/split_stream.py
	# EXECUTION_MODE=shadow_fork TRACE=1 python3 tmp/prime/shadow_execution.py

test:
	pytest --log-cli-level=DEBUG --log-format="%(levelname)s %(process)d %(message)s"


tmp:
	mkdir -p tmp

