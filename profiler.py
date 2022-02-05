
import cProfile
import io
import pstats

def profile(function):
    """
    A decorator that uses CProfile to profile a function
    """

    def inner(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        return_value = function(*args, **kwargs)
        profiler.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats(50)
        print(s.getvalue())
        
        return return_value

    return inner
