import torch
import torch.nn.functional as F




# 1. Hàm kích hoạt Sigmoid có kẹp giá trị chống tràn số exp
def sigmoid(Z):
    # Kẹp Z trong khoảng [-88, 88] để exp không sinh ra inf hay 0
    Z_clamped = torch.clamp(Z, min=-88.0, max=88.0)
    return 1.0 / (1.0 + torch.exp(-Z_clamped))

def grad_sigmoid(A):
    # Đạo hàm của Sigmoid theo đầu ra A: sigma * (1 - sigma)
    return A * (1.0 - A)

# 2. Hàm mất mát Binary Cross Entropy (BCE) cho 40 đặc trưng
def binary_cross_entropy(y_true, y_pred, eps=1e-7):
    # Kẹp xác suất tránh log(0) gây -inf
    y_pred = torch.clamp(y_pred, min=eps, max=1.0 - eps)
    # Tính loss trung bình trên toàn bộ N mẫu và C thuộc tính
    loss = -(y_true * torch.log(y_pred) + (1.0 - y_true) * torch.log(1.0 - y_pred))
    return loss.mean()


def ReLU(Z, alpha = 0.01):
    return torch.where(Z > 0, Z, Z * alpha)

def grad_ReLU(Z, alpha = 0.01):
    return torch.where(Z > 0, torch.ones_like(Z), torch.full_like(Z, alpha))

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