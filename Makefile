.PHONY: all bank test do-2.1

export PYTHONPATH := $(shell pwd):$(PYTHONPATH)

EXAMPLES := $(wildcard examples/*.py)
SUBJECTS := $(notdir $(EXAMPLES:.py=))

all: test $(SUBJECTS) tables

clean:
	rm -rf tmp

$(SUBJECTS): | tmp
	@echo $@
	rm -r tmp/$@ || true
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
	python3 -m pytest --log-cli-level=DEBUG --log-format="%(levelname)s %(process)d %(filename)s:%(lineno)s %(message)s" -vv --tb=short

test-shadow:
	TEST_SKIP_MODES="split,modulo" \
		python3 -m pytest --log-cli-level=DEBUG --log-format="%(levelname)s %(process)d %(filename)s:%(lineno)s %(message)s" -vv

test-split-variants:
	TEST_SKIP_MODES="shadow,shadow_cache,shadow_fork_child,shadow_fork_parent,shadow_fork_cache" \
		python3 -m pytest --log-cli-level=DEBUG --log-format="%(levelname)s %(process)d %(filename)s:%(lineno)s %(message)s" -vv

tmp:
	mkdir -p tmp

tables:
	python3 print_tables.py

