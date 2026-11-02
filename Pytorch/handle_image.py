import torch
import torchvision.transforms.functional as TF



def convert_color(image, color = [1.0, 1.0, 1.0, 1.0]):
    color = torch.tensor(color, dtype = image.dtype, device = image.device)
    color = torch.view(-1, 1, 1)
    image.mul_(color)


def rotate_image(image, angle = 45.0):
    return TF.rotate(image, angle = angle)


def rotate_data(data, label, angles = [15.0, 30.0, 45.0, 60.0, 75.0]):
    mem = [data]
    for i in range(len(angles)):
        mem.append(rotate_image(data, angles[i]))

    if data.ndim == 3:
        return (torch.stack(mem, dim = 0), torch.stack([label] * (len(angles) + 1), dim = 0))
    return (torch.cat(mem, dim = 0), torch.cat([label] * (len(angles) + 1), dim = 0))