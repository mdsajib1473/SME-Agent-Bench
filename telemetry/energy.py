"""GPU energy measurement with NVML.

EnergyMeter samples board power every 100 ms on a background thread and
integrates the samples (trapezoid rule) into watt-hours for the duration of the
with-block. When the driver exposes the cumulative energy counter, its delta is
recorded too as a cross-check. If pynvml or the GPU is unavailable every energy
field stays None and nothing raises.
"""

import threading
import time

SAMPLE_INTERVAL_S = 0.1


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


class EnergyMeter:
    def __init__(self, device_index=0, interval_s=SAMPLE_INTERVAL_S):
        self.device_index = device_index
        self.interval_s = interval_s
        self.available = False
        self.energy_wh = None
        self.counter_energy_wh = None
        self.mean_power_w = None
        self.peak_vram_mib = None
        self.duration_s = None
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
            return
        self.samples.append((time.perf_counter(), watts))
        if self.peak_vram_mib is None or used > self.peak_vram_mib:
            self.peak_vram_mib = used

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
        self.duration_s = time.perf_counter() - self._started
        if not self.available:
            return False
        self._stop.set()
        self._thread.join()
        self._sample()
        counter_end = self._read_counter_mj()
        if self._counter_start is not None and counter_end is not None:
            self.counter_energy_wh = (counter_end - self._counter_start) / 3.6e6
        self._shutdown()

        if len(self.samples) >= 2:
            joules = 0.0
            for (t0, w0), (t1, w1) in zip(self.samples, self.samples[1:]):
                joules += (w0 + w1) / 2.0 * (t1 - t0)
            span = self.samples[-1][0] - self.samples[0][0]
            self.energy_wh = joules / 3600.0
            self.mean_power_w = joules / span if span > 0 else self.samples[0][1]
        return False

    def _shutdown(self):
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        self._nvml = None

    def net_energy_wh(self, idle_power_w):
        """Energy above the idle baseline over the same duration."""
        if self.energy_wh is None or idle_power_w is None:
            return None
        return self.energy_wh - idle_power_w * self.duration_s / 3600.0


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
