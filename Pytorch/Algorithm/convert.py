import torch




LUMA_WEIGHTS = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)

def convert_to_one_hot_coding_1D(device, Y, c = -1):
    if c == -1:
        c = len(torch.unique(Y))
    y = torch.zeros(Y.shape[0], c, device=device)

    for i in range(Y.shape[0]):
        y[i][int(Y[i])] = 1.0
    return y


def convert_to_one_hot_coding(device, Y, c = -1):
    if (len(Y.shape) == 1):
        return convert_to_one_hot_coding_1D(device, Y, c)
    return Y


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