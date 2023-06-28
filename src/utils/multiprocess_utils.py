import multiprocessing

from tqdm import tqdm

_f = None


def _pool_init(f_):
    global _f
    _f = f_


def _pool_executor(i):
    return _f(i)


def run_pool(action, data, n_threads=None):
    """Run Function on each param in Data in parallel Pool
    Args:
        action (function): function to execute
        data (list): list of parameters
        n_threads (int): number of threads, default - number of CPUs (if n_threads == 1 using map w/o Pool)
    Returns:
        list: list of returned values from function
    """
    if n_threads is None:
        n_threads = multiprocessing.cpu_count()
    if n_threads == 1:
        return list(map(action, tqdm(data)))
    else:
        with multiprocessing.Pool(n_threads, _pool_init, (action,)) as p:
            return list(tqdm(p.imap(_pool_executor, data), total=len(data)))
