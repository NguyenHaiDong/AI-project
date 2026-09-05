import time
import torch




def open_gpu():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    return device

def open_cpu():
    device = torch.device("cpu")
    return device




def convert_to_one_hot_coding(device, Y, c = -1):
    if c == -1:
        c = len(torch.unique(Y))
    y = torch.zeros(Y.shape[0], c, device=device)

    for i in range(Y.shape[0]):
        y[i][int(Y[i])] = 1.0
    return y

def get_accuracy_rate(A, B):
    count = 0
    for i in range(len(A)):
        if A[i] == B[i]:
            count += 1
    return (count / len(A)) * 100.0




def ReLU(Z):
    return torch.relu(Z)

def grad_ReLU(Z):
    return (Z > 0).float()

def cost(Y, Y_hat):
    return (-torch.sum(Y * torch.log(Y_hat + 1e-9)) / Y.shape[0])

def softmax(Z):
    max_vals = torch.max(Z, dim = 1, keepdims=True).values
    e_Z = torch.exp(Z - max_vals)
    A = e_Z / torch.sum(e_Z, dim = 1, keepdims=True)
    return A



# Trọng số chuẩn CIE 1931 dạng Tensor
LUMA_WEIGHTS = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)

def convert_to_gray(image):
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)

    # Nếu ảnh đã là 2D (ảnh xám sẵn), chỉ cần duỗi phẳng
    if image.ndim == 2:
        return image.reshape(-1).to(torch.uint8)

    # Đảm bảo trọng số nằm cùng device với tensor ảnh
    weights = LUMA_WEIGHTS.to(image.device)

    # Lấy 3 kênh màu, ép float để nhân ma trận, sau đó đưa về uint8 và làm phẳng
    gray = image[..., :3].to(torch.float32) @ weights
    return gray.to(torch.uint8).reshape(-1)



def open_clock():
    return time.perf_counter()

def close_clock_and_show_time(device, start_time, s = "Tổng thời gian huấn luyện"):
    if device.type == "mps":
        torch.mps.synchronize()

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    print(f"{s}: giấy thứ {elapsed:.3f}")