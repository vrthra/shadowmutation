.PHONY: all bank test do-2.1

export PYTHONPATH := $(shell pwd):$(PYTHONPATH)

EXAMPLES := $(wildcard examples/*.py)
SUBJECTS := $(notdir $(EXAMPLES:.py=))

all: clean test $(SUBJECTS)

clean:
	rm -rf tmp

$(SUBJECTS): | tmp
	@echo $@
	python3 ./ast_mutator.py --ignore "^test_" examples/$@.py tmp/$@
	python execute_versions.py tmp/$@


dev:
	# TRACE=1 python3 tmp/factorization/traditional_32.py
	# EXECUTION_MODE=split GATHER_ATEXIT=1 TRACE=1 python3 tmp/caesar_cypher/split_stream.py
	# EXECUTION_MODE=modulo GATHER_ATEXIT=1 TRACE=1 python3 tmp/caesar_cypher/split_stream.py
	# EXECUTION_MODE=shadow GATHER_ATEXIT=1 python3 tmp/factorization/shadow_execution.py
	EXECUTION_MODE=shadow_cache GATHER_ATEXIT=1 TRACE=1 python3 tmp/factorization/shadow_execution.py
	# EXECUTION_MODE=shadow_fork GATHER_ATEXIT=1 python3 tmp/factorization/shadow_execution.py
	# EXECUTION_MODE=shadow_fork_cache GATHER_ATEXIT=1 TRACE=1 python3 tmp/factorization/shadow_execution.py

test:
	python3 -m pytest --log-cli-level=DEBUG --log-format="%(process)d %(filename)s:%(lineno)s %(levelname)s %(message)s"


tmp:
	mkdir -p tmp

