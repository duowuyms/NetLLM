import cProfile, pstats, io
from pstats import SortKey

class Profiler:
    '''context manager which profiles a block of code,
    then prints out the function calls sorted by cumulative
    execution time
    '''
    def __init__(self, amount=20):
        self.pr = cProfile.Profile()
        self.amount = amount


    def __enter__(self, ):
        self.pr.enable()
        return self


    def __exit__(self):
        self.pr.disable()

        # log stats
        s = io.StringIO()
        ps = pstats.Stats(self.pr, stream=s) \
                   .sort_stats(SortKey.CUMULATIVE)
        ps.print_stats(self.amount)
        print(s.getvalue(), flush=True)