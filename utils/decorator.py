import time
import logging

import traceback

logger = logging.getLogger(__name__)


class TryFunc:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __call__(self, func):
        def inner(*args, **kwargs):
            res = None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                if self.verbose:
                    logger.error(f"in {func.__qualname__}: {(str(traceback.format_exc(chain=1)))}")
                else:
                    logger.error(f"in {func.__qualname__}: {e}")
            finally:
                return res

        return inner


if __name__ == '__main__':
    @TryFunc(True)
    def test():
        print("test")
        a = 1 / 0


    test()
