# Collection of constants used

MAINLINE = 0

# This is used to decide what return values should be untainted when returning from a function.
PRIMITIVE_TYPES = [bool, int, float]


class ShadowExceptionStop(Exception):
    """No more mutants stop this execution."""
    pass


class ShadowException(Exception):
    """Wraps a exception that happened during the run."""
    pass
