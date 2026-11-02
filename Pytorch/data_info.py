import os
from pathlib import Path
import re
from functools import partial
import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.utils as vutils
from mnist import MNIST

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd

# Vô hiệu hóa đa luồng ngầm của OpenCV để tránh tranh chấp luồng trên macOS
cv2.setNumThreads(0)

try:
    import algorithm
except ImportError:
    algorithm = None


def get_MNIST(device, path, number_image=-1):
    mnist = MNIST(path)
    train_data, train_label = mnist.load_training()
    test_data, test_label = mnist.load_testing()

    if number_image > 0:
        train_data = train_data[:number_image]
        train_label = train_label[:number_image]
        test_data = test_data[:number_image]
        test_label = test_label[:number_image]

    # Đẩy thẳng lên GPU/MPS và ép kiểu float32
    train_data = (
        torch.tensor(train_data, dtype=torch.float32, device=device) / 255.0
    )
    test_data = (
        torch.tensor(test_data, dtype=torch.float32, device=device) / 255.0
    )

    train_label = torch.tensor(train_label, dtype=torch.long, device=device)
    test_label = torch.tensor(test_label, dtype=torch.long, device=device)

    return (train_data, train_label, test_data, test_label)


def generate_spiral_data_2D(
    device,
    N=90,
    C=3,
    d=2,
    r_min=0.0,
    r_max=1.0,
    start_degree=0.0,
    number_rotations=1.0,
    noise=0.2,
):
    if d != 2:
        print("Number dimensions is not available now!")

    s = torch.linspace(0.0, 1.0, N, device=device).repeat(C, 1)
    epsilon = torch.randn(C, N, device=device) * noise
    theta_sweep = 2.0 * torch.pi * number_rotations
    j = torch.arange(C, device=device).unsqueeze(1) * (theta_sweep / float(C))

    theta = start_degree + j + theta_sweep * s + epsilon
    r = r_min + (r_max - r_min) * s

    X = torch.zeros((C * N, d), dtype=torch.float32, device=device)
    X[:, 0] = (r * torch.cos(theta)).reshape(-1)
    X[:, 1] = (r * torch.sin(theta)).reshape(-1)

    Y = torch.arange(C, device=device).unsqueeze(1).repeat(1, N).reshape(-1)
    return (X, Y)


def read_image(file_path, width, height):
    try:
        image = cv2.imread(str(file_path))
        if image is None:
            return None
        if image.shape[1] != width or image.shape[0] != height:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(image)
    except Exception:
        return None

def natural_sort_key(p):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", p.name)
    ]


def _load_single_image(args, channels=3, is_flatten=True):
    f, width, height = args
    img = read_image(f, width, height)
    if img is None:
        return None

    # Chuyển về NumPy Array để xử lý kênh màu bằng OpenCV
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.array(img)

    # 1. Xử lý ảnh 1 kênh (Grayscale)
    if channels == 1:
        if arr.ndim == 3 and arr.shape[2] == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            gray = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
        elif arr.ndim == 2:
            gray = arr
        else:
            gray = arr[:, :, 0]

        t = torch.from_numpy(gray)
        # Flatten: [H*W], Không flatten: [1, H, W]
        return t.flatten() if is_flatten else t.unsqueeze(0)

    # 2. Xử lý ảnh 3 kênh (RGB)
    elif channels == 3:
        if arr.ndim == 2:
            rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        else:
            rgb = arr[:, :, :3]

        t = torch.from_numpy(rgb)
        # PyTorch format: chuyển [H, W, C] -> [C, H, W]
        t = t.permute(2, 0, 1)
        # Flatten: [3*H*W], Không flatten: [3, H, W]
        return t.flatten() if is_flatten else t

    else:
        raise ValueError(f"Chỉ hỗ trợ channels = 1 hoặc 3, nhận được: {channels}")


def read_folder(
    folder_path,
    width,
    height,
    number_image=-1,
    channels=3,
    max_workers=8,
    is_flatten=True,
):
    folder = Path(folder_path)
    file_path = [
        f
        for f in folder.iterdir()
        if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
        and not f.name.startswith(".")
    ]
    file_path.sort(key=natural_sort_key)

    total_available = len(file_path)
    if number_image >= 0:
        file_path = file_path[:number_image]

    total_to_read = len(file_path)
    print(
        f"\n[+] Bắt đầu nạp thư mục '{folder.name}': {total_to_read}/{total_available} ảnh (channels={channels})..."
    )

    images = []
    step_log = max(1, total_to_read // 5)
    for idx, f in enumerate(file_path):
        img_t = _load_single_image(
            (f, width, height), channels=channels, is_flatten=is_flatten
        )
        if img_t is not None:
            images.append(img_t)

        current = idx + 1
        if current % step_log == 0 or current == total_to_read:
            percent = (current / total_to_read) * 100
            print(
                f"    -> [{folder.name}] Đã xử lý {current}/{total_to_read} ảnh ({percent:.0f}%)"
            )

    return images


def get_images(
    device,
    folder_paths,
    number_images=-1,
    channels=3,
    width=120,
    height=165,
    is_normalized=True,
    divide=[65.0, 35.0],
    is_shuffle=True,
    manual_label=None,
    is_flatten=True,
    max_workers=8,
):
    if number_images == -1 or isinstance(number_images, int):
        number_images = [number_images] * len(folder_paths)

    all_data = []
    all_label = []

    for i, path in enumerate(folder_paths):
        cur = read_folder(
            path,
            width,
            height,
            number_image=number_images[i],
            channels=channels,
            max_workers=max_workers,
            is_flatten=is_flatten,
        )
        if not cur:
            continue
        all_data.extend(cur)

        lbl = i if manual_label is None else manual_label[i]
        all_label.extend([lbl] * len(cur))
        print(f"✓ Đã nạp xong '{path}': thành công {len(cur)} ảnh.")

    if not all_data:
        raise ValueError("Không tìm thấy ảnh hợp lệ nào trong danh sách thư mục!")

    print("\n[*] Đang tổng hợp Tensor và đồng bộ dữ liệu...")
    data = torch.stack(all_data)
    label = torch.tensor(all_label, dtype=torch.long)

    # Chuẩn hóa về [0.0, 1.0]
    if is_normalized:
        data = data.to(dtype=torch.float32) / 255.0
    else:
        data = data.to(dtype=torch.float32)

    # Xáo trộn dữ liệu trên CPU
    N = data.shape[0]
    if is_shuffle and N > 0:
        perm = torch.randperm(N)
        data = data[perm]
        label = label[perm]

    # Đưa lên thiết bị chỉ định
    data = data.to(device=device)
    label = label.to(device=device)

    # Phân chia Train/Test theo tỷ lệ
    split_idx = int(N * (divide[0] / 100.0))

    print(
        f"✓ Hoàn tất: Train={split_idx} ảnh, Test={N - split_idx} ảnh | Shape: {list(data.shape)} trên '{device}'.\n"
    )
    return (
        data[:split_idx],
        label[:split_idx],
        data[split_idx:],
        label[split_idx:],
    )




def save_images_to_single_folder(
    data: torch.Tensor,
    label: torch.Tensor,
    output_dir: str,
    height: int = 165,
    width: int = 120,
    is_flatten: bool = True,
    is_normalized: bool = True,
    is_shuffle: bool = True,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    data = data.detach().cpu()
    label = label.detach().cpu()
    N = data.shape[0]

    if is_shuffle and N > 0:
        perm = torch.randperm(N)
        data = data[perm]
        label = label[perm]

    if is_normalized:
        data = data * 255.0
    data = data.clamp(0, 255).to(dtype=torch.uint8)

    if is_flatten:
        channels = data.shape[1] // (height * width)
        data = data.view(N, channels if channels in (1, 3) else 1, height, width)
    elif data.ndim == 3:
        data = data.unsqueeze(1)

    for idx in range(N):
        lbl = int(label[idx].item())
        # Lưu định dạng {idx:06d}_{lbl}.png để giữ đúng thứ tự shuffle khi xem thư mục hoặc đọc file
        filename = f"{idx:06d}_{lbl}.png"
        filepath = os.path.join(output_dir, filename)

        img_tensor = data[idx]
        if img_tensor.shape[0] == 1:
            img = Image.fromarray(img_tensor.squeeze(0).numpy(), mode="L")
        else:
            img = Image.fromarray(
                img_tensor.permute(1, 2, 0).numpy(), mode="RGB"
            )
        img.save(filepath)

    print(f"Đã shuffle và lưu thành công {N} ảnh kèm nhãn vào: {output_dir}")


def _read_file(path: Path, width: int, height: int, is_flatten: bool):
    parts = path.stem.split("_")
    # Lấy label từ cuối chuỗi (phù hợp với định dạng {idx:06d}_{lbl})
    lbl = int(parts[-1]) if len(parts) > 1 and parts[-1].lstrip("-").isdigit() else 0

    with Image.open(path) as img:
        if img.size != (width, height):
            img = img.resize((width, height))
        arr = np.array(img)

    t = torch.from_numpy(arr)
    if t.ndim == 2:
        t = t.unsqueeze(0)
    elif t.ndim == 3:
        t = t.permute(2, 0, 1)

    if is_flatten:
        t = t.flatten()

    return t, lbl


def get_images_from_single_folder(
    folder_path: str,
    number_image: int = -1,
    device: str | torch.device = "cpu",
    width: int = 120,
    height: int = 165,
    is_normalized: bool = True,
    divide: list[float] = [65.0, 35.0],
    is_flatten: bool = True,
    max_workers: int = 8,
):
    folder = Path(folder_path)
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    
    file_list = sorted(
        [
            f for f in folder.iterdir()
            if f.suffix.lower() in valid_exts and not f.name.startswith(".")
        ],
        key=natural_sort_key,
    )

    if not file_list:
        raise FileNotFoundError(f"Không có ảnh hợp lệ trong: {folder_path}")

    if number_image > 0:
        file_list = file_list[:number_image]

    results = [_read_file(p, width, height, is_flatten) for p in file_list]

    all_data = [item[0] for item in results]
    # Ép từng label về int thuần túy (tránh trường hợp item[1] bị bọc tensor/mảng con)
    all_label = [
        int(item[1].item() if isinstance(item[1], torch.Tensor) else item[1])
        for item in results
    ]

    data = torch.stack(all_data)
    # Ép phẳng tuyệt đối thành 1D Tensor: shape (N,)
    label = torch.tensor(all_label, dtype=torch.long, device=device).contiguous().view(-1)

    if is_normalized:
        data = data.to(dtype=torch.float32, device=device) / 255.0
    else:
        data = data.to(dtype=torch.float32, device=device)

    N = data.shape[0]
    split_idx = int(N * (divide[0] / 100.0))

    # Cắt train / test và giữ nguyên tensor 1D phẳng
    train_data = data[:split_idx]
    train_label = label[:split_idx].contiguous().view(-1)
    
    test_data = data[split_idx:]
    test_label = label[split_idx:].contiguous().view(-1)

    return (
        train_data,
        train_label,
        test_data,
        test_label,
    )


def save_to_file(nested_lists: list[list[torch.Tensor]], data: list[torch.Tensor], file_name: str) -> None:
    cleaned_mem = [
        [tensor.detach().cpu().contiguous() for tensor in sublist]
        for sublist in nested_lists
    ]
    for tensor in data:
        cleaned_mem.append(tensor.detach().cpu().contiguous())
    torch.save(cleaned_mem, file_name)


def load_from_file(file_name: str, device: str | torch.device = "cpu") -> list[list[torch.Tensor]]:
    target_device = torch.device(device)
    raw_data = torch.load(file_name, map_location=target_device, weights_only=True)

    restored_mem = []
    for item in raw_data:
        if isinstance(item, list):
            restored_mem.append([tensor.to(target_device) for tensor in item])
        elif isinstance(item, torch.Tensor):
            restored_mem.append(item.to(target_device))
    return restored_mem



def load_celeba(
    image_path: str,
    attribute_path: str,
    number_images: int = 10000,
    channels: int = 1,
    labels: list[str] | int | str = "Male",
    image_size: list[int] | tuple[int, int] = [128, 128],
    divide: list[float] = [65.0, 35.0],
    device: str | torch.device = "cpu",
    is_flatten: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tải trực tiếp hình ảnh và nhãn thành Tensor trên device chỉ định,
    phân chia thành tập huấn luyện (training) và tập kiểm tra (testing).

    Tham số:
        is_flatten: Nếu True, làm phẳng mỗi ảnh thành vector 1D [N, channels * H * W].
                    Nếu False, giữ nguyên dạng tensor 4D [N, channels, H, W] cho CNN.

    Trả về:
        training_data, training_labels, testing_data, testing_labels
    """
    target_device = torch.device(device)

    # 1. Đọc dữ liệu bảng nhãn từ tệp CSV và giới hạn số lượng hình ảnh
    dataframe = pd.read_csv(attribute_path)
    if number_images is not None:
        dataframe = dataframe.iloc[:number_images]

    image_names = dataframe["image_id"].values

    # 2. Xử lý danh sách các nhãn thuộc tính (chuyển đổi -1 thành 0)
    all_attribute_columns = [
        column for column in dataframe.columns if column != "image_id"
    ]

    if labels is None:
        selected_labels = all_attribute_columns
    elif isinstance(labels, int):
        selected_labels = all_attribute_columns[:labels]
    elif isinstance(labels, str):
        selected_labels = [labels]
    else:
        selected_labels = list(labels)

    raw_labels = dataframe[selected_labels].to_numpy().copy()
    raw_labels[raw_labels == -1] = 0
    all_labels = torch.tensor(raw_labels, dtype=torch.float32, device=target_device)

    # Nếu chỉ có 1 nhãn thì làm phẳng thành vector một chiều
    if len(selected_labels) == 1:
        all_labels = all_labels.squeeze(dim=-1)

    # 3. Chuỗi biến đổi hình ảnh
    transform_operations = [transforms.Resize(tuple(image_size))]
    if channels == 1:
        transform_operations.append(transforms.Grayscale(num_output_channels=1))
    transform_operations.append(transforms.ToTensor())

    image_transformer = transforms.Compose(transform_operations)

    # 4. Đọc ảnh vào RAM CPU trước, gom thành tensor rồi đẩy lên device đích
    print(f"Đang tải {len(image_names)} hình ảnh lên {target_device}...")
    image_tensor_list = []
    for image_name in image_names:
        image_file_path = os.path.join(image_path, image_name)
        with Image.open(image_file_path) as opened_image:
            converted_image = opened_image.convert("RGB")
            transformed_tensor = image_transformer(converted_image)
            image_tensor_list.append(transformed_tensor)

    all_data = torch.stack(image_tensor_list).to(target_device)

    # Làm phẳng ảnh nếu is_flatten = True: [N, C, H, W] -> [N, C * H * W]
    if is_flatten:
        all_data = all_data.flatten(start_dim=1)

    # 5. Phân chia tập huấn luyện và kiểm tra
    total_samples = len(all_data)
    total_ratio = divide[0] + divide[1]
    training_ratio = divide[0] / total_ratio

    training_count = int(total_samples * training_ratio)

    training_data = all_data[:training_count]
    training_labels = all_labels[:training_count]

    testing_data = all_data[training_count:]
    testing_labels = all_labels[training_count:]

    return training_data, training_labels, testing_data, testing_labels