import torch
import matplotlib.pyplot as plt
import numpy as np

import algorithm



def to_numpy(data):
    """Chuyển đổi an toàn từ PyTorch Tensor (CPU/MPS/CUDA, có/không grad) hoặc List sang NumPy."""
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def show_accuracy_rate_and_number_iterations(device, A, B, it, endline_ = "\n"):
    print(A.shape)
    print(B.shape)
    A = A.to(device)
    B = B.to(device)

    accuracy_rate = algorithm.get_accuracy_rate(A, B)
    # Ép kiểu nếu trả về 0-dim Tensor để in đẹp
    if isinstance(accuracy_rate, torch.Tensor):
        accuracy_rate = accuracy_rate.item()
    if isinstance(it, torch.Tensor):
        it = it.item()

    print(f"Number Iteration(s): {it}", end = endline_)
    print(f"Accuracy rate: {accuracy_rate}%")


def show_image(image, height=None, width=None):
    # 1. Chuyển PyTorch Tensor về NumPy trên CPU
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    elif not isinstance(image, np.ndarray):
        image = np.array(image)

    # 2. Loại bỏ batch dimension nếu ảnh có dạng [1, H, W] hoặc [1, C, H, W]
    if image.ndim == 4 and image.shape[0] == 1:
        image = np.squeeze(image, axis=0)

    # 3. Phục hồi kích thước nếu truyền vào vector 1D (flattened)
    if image.ndim == 1 and height is not None and width is not None:
        channels = image.size // (height * width)
        if channels == 1:
            image = image.reshape(height, width)
        else:
            image = image.reshape(channels, height, width)

    # 4. Xử lý số kênh (Channels)
    # PyTorch format: [C, H, W] -> Chuyển sang Matplotlib format: [H, W, C]
    if image.ndim == 3:
        if image.shape[0] in [1, 3, 4]:  # Kênh nằm ở đầu (C, H, W)
            if image.shape[0] == 1:
                image = np.squeeze(image, axis=0)  # [1, H, W] -> [H, W]
            else:
                image = np.transpose(image, (1, 2, 0))  # [C, H, W] -> [H, W, C]
        elif image.shape[2] == 1:        # Kênh nằm ở cuối [H, W, 1]
            image = np.squeeze(image, axis=2)

    # 5. Chuẩn hóa giá trị điểm ảnh về [0, 1] an toàn cho Matplotlib
    image = image.astype(np.float32)
    min_val, max_val = image.min(), image.max()
    if max_val > 1.0 or min_val < 0.0:
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image)

    # 6. Hiển thị ảnh
    if image.ndim == 2:  # Grayscale
        plt.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    else:               # RGB hoặc RGBA
        plt.imshow(image)

    plt.axis("off")
    plt.show()




def display_clustering_results(
    images,
    true_labels,
    pred_labels,
    cols=25,
    rows_per_class=2,
    figsize=(9, 6),
    dpi=100,
    img_size=(28, 28),
):
    # Ép an toàn cả 3 biến về numpy array
    images = to_numpy(images)
    true_labels = to_numpy(true_labels)
    pred_labels = to_numpy(pred_labels)

    if images.ndim == 2:
        images = images.reshape(-1, img_size[0], img_size[1])

    if images.max() > 1.0:
        images = images / 255.0

    unique_classes = np.unique(pred_labels)
    total_rows = len(unique_classes) * rows_per_class

    cell_h, cell_w = img_size[0] + 2, img_size[1] + 2
    canvas_w = (cols + 1) * cell_w
    canvas_h = total_rows * cell_h
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)

    current_row = 0

    for c_id in unique_classes:
        mask = pred_labels == c_id
        cluster_imgs = images[mask]
        cluster_true = true_labels[mask]

        if len(cluster_imgs) == 0:
            current_row += rows_per_class
            continue

        rep_img = cluster_imgs[0]
        rep_rgb = np.stack([rep_img] * 3, axis=-1)

        max_samples = rows_per_class * cols
        total_samples = min(len(cluster_imgs), max_samples)

        for r_idx in range(rows_per_class):
            y_start = (current_row + r_idx) * cell_h

            # Cột 0: Ô đại diện xanh dương
            if r_idx == 0:
                canvas[y_start : y_start + cell_h, 0:cell_w] = [0.0, 0.4, 1.0]
                canvas[
                    y_start + 1 : y_start + cell_h - 1, 1 : cell_w - 1
                ] = rep_rgb

            # Các cột tiếp theo
            start_i = r_idx * cols
            end_i = min(start_i + cols, total_samples)

            for c_idx, i in enumerate(range(start_i, end_i)):
                img = cluster_imgs[i]
                is_error = cluster_true[i] != c_id
                x_start = (c_idx + 1) * cell_w

                if is_error:
                    img_rgb = np.zeros(
                        (img_size[0], img_size[1], 3), dtype=np.float32
                    )
                    img_rgb[:, :, 0] = 0.85
                    img_rgb[img > 0.2] = [1.0, 1.0, 1.0]
                    border_color = [0.85, 0.0, 0.0]
                else:
                    img_rgb = np.stack([img] * 3, axis=-1)
                    border_color = [0.15, 0.15, 0.15]

                canvas[
                    y_start : y_start + cell_h, x_start : x_start + cell_w
                ] = border_color
                canvas[
                    y_start + 1 : y_start + cell_h - 1,
                    x_start + 1 : x_start + cell_w - 1,
                ] = img_rgb

        current_row += rows_per_class

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor="black")
    ax.imshow(canvas, aspect="equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()




def plot_spiral_data(
    X: torch.Tensor,
    y: torch.Tensor,
    title: str = "Tập dữ liệu Spiral",
    figsize: tuple = (7, 7),
    point_size: int = 40,
    cmap: str = "Spectral",
) -> None:
    """Chuyển đổi tensor PyTorch từ GPU về CPU và trực quan hóa bằng scatter

    plot.
    """
    # 1. Chuyển tensor từ GPU về CPU và sang NumPy
    # Dùng detach() để phòng trường hợp tensor có gắn autograd graph
    X_np = X.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    # 2. Khởi tạo khung vẽ
    fig, ax = plt.subplots(figsize=figsize)

    # 3. Vẽ scatter plot
    scatter = ax.scatter(
        X_np[:, 0],
        X_np[:, 1],
        c=y_np,
        cmap=cmap,
        s=point_size,
        edgecolors="k",
        linewidths=0.6,
        alpha=0.85,
    )

    # 4. Thêm thanh chú thích màu (Colorbar) theo từng class
    cbar = plt.colorbar(scatter, ax=ax, ticks=torch.unique(y).cpu().numpy())
    cbar.set_label("Nhãn lớp (Class ID)")

    # 5. Định dạng hệ trục
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel(r"$X_1$", fontsize=11)
    ax.set_ylabel(r"$X_2$", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.axis("equal")  # Giữ tỷ lệ 1:1 tránh méo hình

    plt.tight_layout()
    plt.show()

    # Cách dùng:
    # X, y = generate_spiral_data(N=150, C=3, noise=0.15)
    # plot_spiral_data(X, y)



def show_image_and_attribute(image, width, height, label, attributes):
    print("\n============================= ATTRIBUTE(S) =============================\n")
    for i in range(label.shape[0]):
        if (label[i] == 1):
            print(attributes[i])
    print("\n========================================================================\n")
    show_image(image, width, height)