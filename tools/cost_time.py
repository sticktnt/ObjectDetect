import time


def cost_time(func):
    """
    装饰器 统计函数执行时间
    """
    def fun(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        end_t = time.perf_counter()
        print(f'func {func.__name__} cost time:{end_t - t:.8f} s')
        with open(r'D:\work\python\ObiectDetect\tests\test_time.csv','a') as f:
            f.write(f'{func.__name__},{end_t - t:.8f}\n')
        return result

    return fun
