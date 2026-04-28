import time
from datetime import timedelta
from functools import wraps

def time_this(func):
    @wraps(func)
    def wrapper(*arg, **kw):
        should_time = kw.pop('time', False)
        if should_time:
            t1 = time.time()
            res = func(*arg, **kw)
            t2 = time.time()
            t = str(timedelta(seconds=round(t2-t1, 4)))

            # check if the name is __init__ and if it is, print the class name and 
            # the time it took to create the object
            if func.__name__ == "__init__":
                print(f"\t>>{arg[0].__class__.__name__}.{func.__name__} took {t}")
            else:
                print(f"\t>>{func.__name__} took {t}")
            return res
        else:
            return func(*arg, **kw)
    return wrapper

# TODO add the parameter count function 