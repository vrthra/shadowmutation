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
	# TRACE=1 python3 tmp/approx_exp/traditional_33.py
	# EXECUTION_MODE=split GATHER_ATEXIT=1 TRACE=1 python3 tmp/caesar_cypher/split_stream.py
	# EXECUTION_MODE=modulo GATHER_ATEXIT=1 TRACE=1 python3 tmp/caesar_cypher/split_stream.py
	# EXECUTION_MODE=shadow GATHER_ATEXIT=1 TRACE=1 python3 tmp/caesar_cypher/shadow_execution.py
	# EXECUTION_MODE=shadow_cache GATHER_ATEXIT=1 TRACE=1 python3 tmp/prime/shadow_execution.py
	EXECUTION_MODE=shadow_fork GATHER_ATEXIT=1 TRACE=1 python3 tmp/approx_exp/shadow_execution.py
	# EXECUTION_MODE=shadow_fork_cache GATHER_ATEXIT=1 TRACE=1 python3 tmp/approx_exp/shadow_execution.py

test:
	python3 -m pytest --log-cli-level=DEBUG --log-format="%(levelname)s %(process)d %(filename)s:%(lineno)s %(message)s"

test-shadow:
	TEST_SKIP_SPLIT_MODES=1 \
		python3 -m pytest --log-cli-level=DEBUG --log-format="%(levelname)s %(process)d %(filename)s:%(lineno)s %(message)s"

test-shadow-fork:
	TEST_SKIP_SPLIT_MODES=1 TEST_SKIP_SHADOW_NO_FORK=1 \
		python3 -m pytest --log-cli-level=DEBUG --log-format="%(levelname)s %(process)d %(filename)s:%(lineno)s %(message)s"

test-shadow-no-fork:
	TEST_SKIP_SPLIT_MODES=1 TEST_SKIP_SHADOW_FORK=1 \
		python3 -m pytest --log-cli-level=DEBUG --log-format="%(levelname)s %(process)d %(filename)s:%(lineno)s %(message)s"


tmp:
	mkdir -p tmp

