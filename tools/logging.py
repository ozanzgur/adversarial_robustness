import contextlib, sys
"""
https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python

"""

@contextlib.contextmanager
def log_print(file):
    # capture all outputs to a log file while still printing it
    class Logger:
        def __init__(self, file):
            self.terminal = sys.stdout
            self.log = file

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def __getattr__(self, attr):
            return getattr(self.terminal, attr)

    logger = Logger(file)

    _stdout = sys.stdout
    _stderr = sys.stderr
    sys.stdout = logger
    sys.stderr = logger
    try:
        yield logger.log
    finally:
        sys.stdout = _stdout
        sys.stderr = _stderr