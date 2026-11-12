import torch
import psutil
import os




def open_gpu():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    return device


def open_multi_gpu(num_devices=None):
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Tìm thấy {gpu_count} GPU CUDA.")

        # Nếu không truyền số lượng: trả về danh sách tất cả GPU hiện có
        if num_devices is None:
            return [torch.device(f"cuda:{i}") for i in range(gpu_count)]

        # Nếu truyền số lượng: tự chia vòng lặp (ví dụ 4 model trên 2 GPU -> cuda:0, cuda:1, cuda:0, cuda:1)
        return [torch.device(f"cuda:{i % gpu_count}") for i in range(num_devices)]

    elif torch.backends.mps.is_available():
        print("Chạy trên Apple Silicon (MPS).")
        dev = torch.device("mps")
        n = num_devices if num_devices is not None else 1
        return [dev] * n

    else:
        print("Không có GPU, chuyển sang CPU.")
        dev = torch.device("cpu")
        n = num_devices if num_devices is not None else 1
        return [dev] * n


def open_cpu():
    device = torch.device("cpu")
    return device


def get_ram_usage():
    # Lấy thông tin tiến trình hiện tại
    process = psutil.Process(os.getpid())
    # rss (Resident Set Size): lượng RAM vật lý thực tế đang chiếm
    mem_bytes = process.memory_info().rss
    mem_mb = mem_bytes / (1024 ** 2)
    return mem_mb