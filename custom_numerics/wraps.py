import functools
import time


def timer(func):
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        runtime = time.perf_counter() - start
        print(f"{func.__name__} took {runtime//60} min {runtime%60:.4f} secs")
        if func.__name__ == "Compute":
            return result, runtime
        return result

    return _wrapper
