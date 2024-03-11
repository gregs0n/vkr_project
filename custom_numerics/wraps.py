import functools
import time


def timer(func):
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        runtime = time.perf_counter() - start
        line = f"{func.__name__} took {runtime//60} min {runtime%60:.4f} secs"
        if func.__name__ in ("Compute", "BiCGstab"):
            args[0]._log(line)
        else:
            print(f"{time.strftime('%H:%M:%S')} - " + line)
        return result

    return _wrapper
