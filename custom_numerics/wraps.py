import functools
import time


def timer(func):
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        runtime = time.perf_counter() - start
        line = f"{func.__name__} took {runtime//60} min {runtime%60:.4f} secs"
        if func.__name__ == "_compute" or "BiCGstab":
            args[0]._log(line)
        else:
            print(line)
        return result

    return _wrapper
