.PHONY: all clean short long

export PYTHONPATH := $(shell pwd):$(PYTHONPATH)

EXAMPLES_LONG := $(wildcard examples_long/*.py)
EXAMPLES_SHORT := $(wildcard examples_short/*.py)
SUBJECTS_LONG := $(EXAMPLES_LONG:.py=)
SUBJECTS_SHORT := $(EXAMPLES_SHORT:.py=)

all: test short long tables

ifndef VERBOSE
.SILENT:
endif

clean:
	rm -rf tmp

short: $(SUBJECTS_SHORT)

$(SUBJECTS_SHORT): PROG=$(notdir $@)
$(SUBJECTS_SHORT): | tmp
	@echo "$@ -> $(PROG)"
	rm -r tmp/short/$(PROG) || true
	python3 ./ast_mutator.py --ignore "^test_" examples_short/$(PROG).py tmp/short/$(PROG)
	python execute_versions.py tmp/short/$(PROG)

long: $(SUBJECTS_LONG)

$(SUBJECTS_LONG): PROG=$(notdir $@)
$(SUBJECTS_LONG): | tmp
	@echo "$@ -> $(PROG)"
	rm -r tmp/long/$(PROG) || true
	python3 ./ast_mutator.py --ignore "^test_" examples_long/$(PROG).py tmp/long/$(PROG)
	EXEC_NO_TRACE=1 python execute_versions.py tmp/long/$(PROG)

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
