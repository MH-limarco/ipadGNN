"""
這個模組提供了 Timer 與 MultiTimer 兩個類別，用於計時與批量管理計時器。
"""

__all__ = ["Timer", "MultiTimer"]


import time

timer_func = time.perf_counter

def current_time():
    return timer_func()


class MultiTimer:
    def __init__(self):
        self.timers = {}

    def start(self, name):
        if name not in self.timers:
            self.timers[name] = Timer()
        self.timers[name].start()

    def stop(self, name):
        if name not in self.timers:
            raise RuntimeError(f"Timer-{name} not found")
        self.timers[name].stop()

    def start_all(self, names):
        """ 批量同步啟動計時器 """
        reference_time = current_time()
        for name in names:
            if name not in self.timers:
                self.timers[name] = Timer()
            self.timers[name].start(reference_time)

    def stop_all(self, names=None):
        """ 批量同步停止計時器 """
        errors = []
        names = names or list(self.timers.keys())
        reference_time = current_time()
        for name in names:
            if name not in self.timers:
                errors.append(f"Timer '{name}' not found")
                continue
            try:
                self.timers[name].stop(reference_time)
            except RuntimeError as e:
                errors.append(f"Timer '{name}': {e}")
        if errors:
            raise RuntimeError("\n".join(errors))


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        """ 重置計時器 """
        self._start_time = None
        self._end_time = None

    def start(self, reference_time=None):
        """ 開始計時 """
        if self._start_time is not None:
            raise RuntimeError("Timer already started")
        self._start_time = current_time() if reference_time is None else reference_time

    def stop(self, reference_time=None):
        """ 停止計時 """
        if self._start_time is None:
            raise RuntimeError("Timer has not been started.")
        if self._end_time is not None:
            raise RuntimeError("Timer has already been stopped.")
        self._end_time = current_time() if reference_time is None else reference_time

    def get_time(self):
        """ 獲取計時結果  """
        if self._start_time is None:
            raise RuntimeError("Timer has not been started.")
        if self._end_time is None:
            raise RuntimeError("Timer is still running. Call stop() first.")
        return self._end_time - self._start_time

    def set_start_time(self, start_time):
        self._start_time = start_time

    def set_end_time(self, end_time):
        self._end_time = end_time

    @property
    def is_running(self):
        """檢查計時器是否正在運行"""
        return self._start_time is not None and self._end_time is None
