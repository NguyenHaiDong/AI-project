import torch
import Algorithm.algorithm as algorithm
import torch.nn.functional as F
import data_info




def extract_for_MLP(
    device, X, Y, number_neurons_per_layer = [100, 100], c = -1,
    is_need_to_convert_one_hot_coding = False, file_pretraining = "None",
    pre_W = None, pre_B = None
):
    n = X.shape[0]
    d = X.shape[1]
    if c == -1:
        c = torch.unique(Y).numel()
    if is_need_to_convert_one_hot_coding == True:
        Y = algorithm.convert_to_one_hot_coding(device, Y, c)

    dims = [d] + number_neurons_per_layer + [c]
    number_layers = len(dims) - 1

    if file_pretraining == "None":
        W = [torch.randn(dims[i], dims[i + 1], device=device) * algorithm.get_better_randn(dims[i]) for i in range(number_layers)]
        B = [torch.zeros(1, dims[i + 1], device=device) for i in range(number_layers)]
    elif file_pretraining == "CNNs":
        W, B = pre_W, pre_B
    else:
        W, B = data_info.load_from_file(file_pretraining, device = device)

    grad_W = [torch.zeros_like(w, device=device) for w in W]
    grad_B = [torch.zeros_like(b, device=device) for b in B]

    A = [None for _ in range(number_layers + 1)]
    Z = [None for _ in range(number_layers)]

    return (W, B, A, Z, grad_W, grad_B, n, d, c, number_layers, file_pretraining)
