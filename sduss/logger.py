"""Logging configuration"""

import logging
import sys

_root_logger = logging.getLogger("sduss")
_default_handler = None
_DEFAULT_ROOT_LOGGER_LEVEL = logging.DEBUG
_DEFAULT_HANDLER_LEVEL = logging.DEBUG

_FORMAT = "%(levelname)s %(asctime)s %(filename)s: %(lineno)d %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"

class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""
    
    def __init__(self, fmt, datefmt = None):
        super().__init__(fmt=fmt, datefmt=datefmt)
        
    def format(self, record):
        msg: str = super().format(record)
        header = msg.split(record.message)[0]
        msg = msg.replace('\n', '\n' + header)
        return msg
        

def _setup_logger():
    _root_logger.setLevel(_DEFAULT_ROOT_LOGGER_LEVEL)
    global _default_handler
    if _default_handler is None:
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.flush = sys.stdout.flush  # bind the flush method
        _default_handler.setLevel(_DEFAULT_HANDLER_LEVEL)
        _root_logger.addHandler(_default_handler)
    
    formatter = NewLineFormatter(fmt=_FORMAT, datefmt=_DATE_FORMAT)
    _default_handler.setFormatter(formatter)
    
    # stop the message from being propagated to the parent logger
    _root_logger.propagate = False
    

def init_logger(name: str):
    return logging.getLogger(name)

# setup the logger when the module is imported
_setup_logger()
_root_logger.info("setting up logger")