import time
import torch




def open_clock():
    return time.perf_counter()

def close_clock_and_show_time(device, start_time, s = "Tổng thời gian huấn luyện"):
    if device.type == "mps":
        torch.mps.synchronize()

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    print(f"{s}: giấy thứ {elapsed:.3f}")