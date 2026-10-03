import time
import torch
import torch.nn.functional as F




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

def max_pooling(A, kernel_size = 3, stride = 3, padding = 0):
    N, C, H, W = A.shape
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1

    A_unfold = F.unfold(A, kernel_size = kernel_size, stride = stride, padding = padding)
    A_unfold = A_unfold.view(N, C, kernel_size * kernel_size, H_out * W_out)

    values, indices = torch.max(A_unfold, dim = 2)

    A_unfold = values
    A_unfold = A_unfold.view(N, C, H_out, W_out)

    return [A_unfold, [indices, A.shape, kernel_size, stride, padding]]

def grad_max_pooling(dout, cache):
    indices, shape, kernel_size, stride, padding = cache
    N, C, H, W = shape

    A_flat = dout.view(N, C, 1, -1)

    A_unfold = torch.zeros(N, C, kernel_size * kernel_size, A_flat.shape[-1], dtype = dout.dtype, device = dout.device)
    A_unfold.scatter_(dim = 2, index = indices.unsqueeze(2), src = A_flat)
    A_unfold = A_unfold.view(N, C * kernel_size * kernel_size, A_unfold.shape[-1])

    return F.fold(A_unfold, output_size = (H, W), kernel_size = kernel_size, stride = stride, padding = padding)




# Trọng số chuẩn CIE 1931 dạng Tensor
LUMA_WEIGHTS = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)

def convert_to_gray(image, is_flatten = True):
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)

    # Đảm bảo trọng số nằm cùng device với tensor ảnh
    weights = LUMA_WEIGHTS.to(image.device)

    # Lấy 3 kênh màu, ép float để nhân ma trận, sau đó đưa về uint8 và làm phẳng
    gray = image[..., :3].to(torch.float32) @ weights

    if is_flatten == True:
        return gray.to(torch.uint8).reshape(-1)
    return gray.unsqueeze(0)



def open_clock():
    return time.perf_counter()

def close_clock_and_show_time(device, start_time, s = "Tổng thời gian huấn luyện"):
    if device.type == "mps":
        torch.mps.synchronize()

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    print(f"{s}: giấy thứ {elapsed:.3f}")




def get_better_randn(n):
    return (2.0 / n) ** 0.5

class Adam:
    def __init__(self, device, data, beta1 = 0.9, beta2 = 0.999, eta = 3e-4, epsilon = 1e-7, weight_decay = 0.01):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        self.number_layers = len(data[0])
        self.b1 = 1.0
        self.b2 = 1.0

        self.W_M = [torch.zeros_like(w, device=device) for w in data[0]]
        self.W_V = [torch.zeros_like(w, device=device) for w in data[0]]

        self.B_M = [torch.zeros_like(b, device=device) for b in data[1]]
        self.B_V = [torch.zeros_like(b, device=device) for b in data[1]]


    def gradient_descent(self, it, W, B, grad_W, grad_B):
        self.b1 *= self.beta1
        self.b2 *= self.beta2

        b1 = 1.0 - self.b1
        b2 = 1.0 - self.b2
        a = self.eta * (b2 ** 0.5) / b1
        b = self.epsilon * (b2 ** 0.5)
        decay_factor = 1.0 - self.eta * self.weight_decay

        for i in range(self.number_layers):
            self.W_M[i].mul_(self.beta1).add_(grad_W[i], alpha = (1.0 - self.beta1))
            self.W_V[i].mul_(self.beta2).addcmul_(grad_W[i], grad_W[i], value = (1.0 - self.beta2))

            self.B_M[i].mul_(self.beta1).add_(grad_B[i], alpha = (1.0 - self.beta1))
            self.B_V[i].mul_(self.beta2).addcmul_(grad_B[i], grad_B[i], value = (1.0 - self.beta2))

            # W[i].mul_(decay_factor)

            denom_W = torch.sqrt(self.W_V[i]).add_(b)
            W[i].addcdiv_(self.W_M[i], denom_W, value = -a)

            denom_B = torch.sqrt(self.B_V[i]).add_(b)
            B[i].addcdiv_(self.B_M[i], denom_B, value = -a)




class Momentum:
    def __init__(self, device, data, eta = 0.01, gamma = 0.9, decay_rate = 1e-3, weight_decay = 0.01):
        self.eta = eta
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.weight_decay = weight_decay

        self.number_layers = len(data[0])
        self.V_W = [torch.zeros_like(w, device=device) for w in data[0]]
        self.V_B = [torch.zeros_like(b, device=device) for b in data[1]]


    def gradient_descent(self, it, W, B, grad_W, grad_B):
        cur_eta = self.eta / (1.0 + self.decay_rate * it)
        decay_factor = 1.0 - cur_eta * self.weight_decay
        eta_gamma = cur_eta * self.gamma

        for i in range(self.number_layers):
            self.V_W[i].mul_(self.gamma).add_(grad_W[i])
            self.V_B[i].mul_(self.gamma).add_(grad_B[i])

            W[i].mul_(decay_factor)
            W[i].add_(grad_W[i], alpha = -cur_eta).add_(self.V_W[i], alpha = -eta_gamma)
            B[i].add_(grad_B[i], alpha = -cur_eta).add_(self.V_B[i], alpha = -eta_gamma)