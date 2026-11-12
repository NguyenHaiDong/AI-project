import Algorithm.algorithm as algorithm
import torch
import torch.nn.functional as F
import MLPs.mlps as mlp
import display
import gc
import handle_image as handle_image
import data_info




def prepare_for_extract(
    device, X, Y, number_conv_layer, c, p, s,
    kernel, max_pooling_activate, max_pooling_padding, max_pooling_kernel_size, max_pooling_stride, C
):
    X = X.to(device)
    n = X.shape[0]
    if c == -1:
        c = torch.unique(Y).numel()
    Y = algorithm.convert_to_one_hot_coding(device, Y, c)

    h = X.shape[2]
    w = X.shape[3]
    sz = []
    for i in range(number_conv_layer):
        cur_h = (h + 2 * p - kernel[i]) // s + 1
        cur_w = (w + 2 * p - kernel[i]) // s + 1
        sz.append([cur_h, cur_w])

        if max_pooling_activate[i] == 1:
            cur_h = (cur_h + 2 * max_pooling_padding - max_pooling_kernel_size) // max_pooling_stride + 1
            cur_w = (cur_w + 2 * max_pooling_padding - max_pooling_kernel_size) // max_pooling_stride + 1
        h = cur_h
        w = cur_w
    dims = [X.shape[1]] + C
    return (X, n, Y, h, w, sz, dims)


def build_matrices(device, file_pretraining, dims, kernel, number_conv_layer):
    if file_pretraining == "None":
        W = [torch.randn(dims[i + 1], dims[i] * kernel[i] * kernel[i], device = device) * algorithm.get_better_randn(dims[i] * kernel[i] * kernel[i]) for i in range(number_conv_layer)]
        B = [torch.zeros(dims[i + 1], 1, device = device) for i in range(number_conv_layer)]
        grad_W = [torch.zeros_like(w) for w in W]
        grad_B = [torch.zeros_like(b) for b in B]
        mem_pretrainning = None
    else:
        mem_pretrainning = data_info.load_from_file(file_pretraining, device = device)
        # print(mem_pretrainning[-1])
        W = mem_pretrainning[0]
        B = mem_pretrainning[1]
        grad_W = [torch.zeros_like(w) for w in mem_pretrainning[0]]
        grad_B = [torch.zeros_like(b) for b in mem_pretrainning[1]]

    A = [None for _ in range(number_conv_layer + 1)]
    Z = [None for _ in range(number_conv_layer)]
    return (W, B, grad_W, grad_B, mem_pretrainning, A, Z)


def extract_for_CNN(
    device, X, Y, layer_size_for_mlp = [100, 100], c = -1, list_func = None,
    number_conv_layer = 3, C = [16, 16, 16], kernel = [5, 5, 5], s = 1, p = 1,
    max_pooling_kernel_size = 3, max_pooling_stride = 3, max_pooling_padding = 0,
    max_pooling_activate = [1, 1, 1], has_GAP = False, has_batch_normalization = False,
    file_pretraining = "None"
):
    X, n, Y, h, w, sz, dims = prepare_for_extract(
        device, X, Y, number_conv_layer, c, p, s,
        kernel, max_pooling_activate, max_pooling_padding, max_pooling_kernel_size, max_pooling_stride, C
    )
    W, B, grad_W, grad_B, mem_pretraining, A, Z = build_matrices(device, file_pretraining, dims, kernel, number_conv_layer)

    return [
        n, Y, W, B, grad_W, grad_B, A, Z, dims, sz, c, number_conv_layer, s, p, C, kernel, h, w, layer_size_for_mlp, 
        max_pooling_kernel_size, max_pooling_stride, max_pooling_padding,
        max_pooling_activate, has_GAP, has_batch_normalization, 
        file_pretraining, mem_pretraining
    ]