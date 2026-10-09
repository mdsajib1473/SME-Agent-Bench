"""GPU energy measurement with NVML.

EnergyMeter samples board power every 100 ms on a background thread for the
duration of the with-block. On this GPU (Ampere) the reported power is a 1 s
average refreshed about every 0.5 s, so it trails the true power by about
0.65 s. With lag_s > 0 the sample timestamps are shifted back by lag_s, the
thread keeps sampling for LAG_TAIL_S after the block so the shifted series
covers the whole block, and the shifted series is integrated over the block
only (trapezoid rule). That lag corrected figure is energy_wh; the unshifted
integral over the block is energy_raw_wh. When the driver exposes the
cumulative energy counter, its delta over the block is recorded too, as a
cross-check. If pynvml, the GPU or the counter is unavailable the affected
fields stay None and nothing raises.
"""

import threading
import time

SAMPLE_INTERVAL_S = 0.1
LAG_TAIL_S = 1.0


def _load_nvml():
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    return pynvml


def integrate_wh(samples, start, end, lag_s=0.0):
    """Trapezoid energy in Wh of (time, watts) samples shifted back by lag_s, over [start, end].

    Segments that cross a window edge are cut there with linear interpolation.
    """
    points = sorted((t - lag_s, w) for t, w in samples)
    joules = 0.0
    for (t0, w0), (t1, w1) in zip(points, points[1:]):
        a, b = max(t0, start), min(t1, end)
        if b <= a:
            continue
        slope = (w1 - w0) / (t1 - t0)
        joules += (w0 + slope * (a - t0) + w0 + slope * (b - t0)) / 2.0 * (b - a)
    return joules / 3600.0


class EnergyMeter:
    def __init__(self, device_index=0, interval_s=SAMPLE_INTERVAL_S, lag_s=0.0):
        self.device_index = device_index
        self.interval_s = interval_s
        self.lag_s = lag_s
        self.available = False
        self.energy_wh = None
        self.energy_raw_wh = None
        self.counter_energy_wh = None
        self.mean_power_w = None
        self.peak_vram_mib = None
        self.duration_s = None
        self.ended = None
        self.samples = []
        self._nvml = None
        self._handle = None
        self._stop = threading.Event()
        self._thread = None
        self._started = None
        self._counter_start = None

    def _sample(self):
        try:
            watts = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            used = self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / 1024**2
        except Exception:
            return None
        sample = (time.perf_counter(), watts)
        self.samples.append(sample)
        if self.peak_vram_mib is None or used > self.peak_vram_mib:
            self.peak_vram_mib = used
        return sample

    def _loop(self):
        while not self._stop.wait(self.interval_s):
            self._sample()

    def _read_counter_mj(self):
        try:
            return self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
        except Exception:
            return None

    def __enter__(self):
        self._started = time.perf_counter()
        self._nvml = _load_nvml()
        if self._nvml is not None:
            try:
                self._handle = self._nvml.nvmlDeviceGetHandleByIndex(self.device_index)
                self.available = True
            except Exception:
                self._shutdown()
        if self.available:
            self._counter_start = self._read_counter_mj()
            self._sample()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.ended = time.perf_counter()
        self.duration_s = self.ended - self._started
        if not self.available:
            return False
        last = self._sample()
        counter_end = self._read_counter_mj()
        if self.lag_s > 0:
            time.sleep(LAG_TAIL_S)
        self._stop.set()
        self._thread.join()
        if self._counter_start is not None and counter_end is not None:
            self.counter_energy_wh = (counter_end - self._counter_start) / 3.6e6
        self._shutdown()

        self.samples.sort()
        start = self.samples[0][0] if self.samples else None
        end = last[0] if last is not None else self.ended
        block = [s for s in self.samples if s[0] <= end]
        if len(block) >= 2:
            self.energy_raw_wh = integrate_wh(block, start, end)
            span = end - start
            self.mean_power_w = self.energy_raw_wh * 3600.0 / span if span > 0 else block[0][1]
            self.energy_wh = (
                integrate_wh(self.samples, start, end, self.lag_s) if self.lag_s > 0 else self.energy_raw_wh
            )
        return False

    def _shutdown(self):
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        self._nvml = None

    def net_energy_wh(self, idle_power_w):
        """Sampled energy (lag corrected when lag_s > 0) above the idle baseline over the same duration."""
        if self.energy_wh is None or idle_power_w is None:
            return None
        return self.energy_wh - idle_power_w * self.duration_s / 3600.0

    def net_counter_energy_wh(self, idle_power_w):
        """Counter energy above the idle baseline over the same duration."""
        if self.counter_energy_wh is None or idle_power_w is None:
            return None
        return self.counter_energy_wh - idle_power_w * self.duration_s / 3600.0


def gpu_info(device_index=0):
    """GPU name, total memory and driver version, or None without a GPU."""
    nvml = _load_nvml()
    if nvml is None:
        return None
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(device_index)
        name = nvml.nvmlDeviceGetName(handle)
        driver = nvml.nvmlSystemGetDriverVersion()
        return {
            "name": name.decode("utf-8") if isinstance(name, bytes) else name,
            "total_mib": nvml.nvmlDeviceGetMemoryInfo(handle).total / 1024**2,
            "driver": driver.decode("utf-8") if isinstance(driver, bytes) else driver,
        }
    except Exception:
        return None
    finally:
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass


def measure_idle_power(seconds=30, device_index=0):
    """Mean board power in watts while nothing runs, or None without a GPU."""
    with EnergyMeter(device_index=device_index) as meter:
        if meter.available:
            time.sleep(seconds)
    return meter.mean_power_w
