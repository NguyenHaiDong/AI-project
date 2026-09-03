from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import cv2
from mnist import MNIST
import torch
import algorithm


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
    image = cv2.imread(str(file_path))
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    # Trả về PyTorch Tensor CPU dạng uint8
    return torch.from_numpy(image)


def natural_sort_key(p):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", p.name)
    ]


def _load_single_image(args):
    f, width, height = args
    img = read_image(f, width, height)
    return algorithm.convert_to_gray(img)


def read_folder(folder_path, width, height, number_image=-1, max_workers=8):
    folder = Path(folder_path)
    file_path = [
        f
        for f in folder.iterdir()
        if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]
        and not f.name.startswith(".")
    ]
    file_path.sort(key=natural_sort_key)

    if number_image >= 0:
        file_path = file_path[:number_image]

    tasks = [(f, width, height) for f in file_path]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        images = list(executor.map(_load_single_image, tasks))

    return images


def get_images(
    device,
    folder_paths,
    number_images=-1,
    width=120,
    height=165,
    is_normalized=True,
    divide=[65.0, 35.0],
    is_shuffle=True,
    manual_label=None,
    flatten=True,
    max_workers=8,
):
    if number_images == -1 or isinstance(number_images, int):
        number_images = [number_images] * len(folder_paths)

    all_data = []
    all_label = []

    for i, path in enumerate(folder_paths):
        cur = read_folder(
            path, width, height, number_images[i], max_workers=max_workers
        )
        if not cur:
            continue
        all_data.extend(cur)

        lbl = i if manual_label is None else manual_label[i]
        all_label.extend([lbl] * len(cur))

    # 1. Ghép danh sách tensor bằng PyTorch thuần (không qua NumPy)
    data = torch.stack(all_data)
    label = torch.tensor(all_label, dtype=torch.long)

    # 2. Đẩy thẳng toàn bộ lên GPU/MPS
    data = data.to(device=device)
    label = label.to(device=device)

    # 3. Chuẩn hóa ma trận trực tiếp trên GPU
    if is_normalized:
        data = data.to(dtype=torch.float32) / 255.0
    else:
        data = data.to(dtype=torch.float32)

    if flatten:
        data = data.view(data.shape[0], -1)

    # 4. Shuffle trực tiếp trên GPU bằng torch.randperm
    N = data.shape[0]
    if is_shuffle:
        perm = torch.randperm(N, device=device)
        data = data[perm]
        label = label[perm]

    # 5. Phân chia Train/Test view
    split_idx = int(N * (divide[0] / 100.0))

    return (
        data[:split_idx],
        label[:split_idx],
        data[split_idx:],
        label[split_idx:],
    )