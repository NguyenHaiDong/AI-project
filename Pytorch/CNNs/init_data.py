import algorithm
import torch
import torch.nn.functional as F
import MLPs as mlp
import display
import gc
import handle_image as handle_image
import data_info




def build_bns(mem_pretraining, file_pretraining, has_batch_normalization, number_conv_layer):
    if file_pretraining != "None":
        pre_data = mem_pretraining[4:]
        count_it = pre_data[0].item()
        step_count = pre_data[1].item()

        index = 2
        if has_batch_normalization == True:
            bns = [algorithm.Batch_Normalization(var_run = pre_data[index + i * 2], mu_run = pre_data[index + i * 2 + 1]) for i in range(number_conv_layer)] 
    else:
        count_it = 0
        step_count = 0

        if has_batch_normalization == True:
            bns = [algorithm.Batch_Normalization() for _ in range(number_conv_layer)] 
    return (bns, count_it, step_count)


def build_MLP(device, mem_pretraining, file_pretraining, layer_size_for_mlp, has_GAP, final_h, final_w, drop_out, Y, C, c, n):
    size_for_X_in_MLP = C[-1]
    if has_GAP == False:
        size_for_X_in_MLP *= final_h * final_w

    file_pretraining_for_mlp = None
    if file_pretraining == "None":
        file_pretraining_for_mlp = "None"
    else:
        file_pretraining_for_mlp = "CNNs"

    mlp_data = mlp.extract_for_MLP(
        device, torch.zeros(n, size_for_X_in_MLP, device = device), Y, 
        number_neurons_per_layer = layer_size_for_mlp, 
        list_func = [
            algorithm.ReLU, algorithm.grad_ReLU,
            algorithm.sigmoid, algorithm.binary_cross_entropy
        ],
        is_need_to_convert_one_hot_coding = False,
        file_pretraining = file_pretraining_for_mlp,
        c = c
    )

    WB = None
    W = B = None
    if file_pretraining == "None":
        WB = (mlp_data[1], mlp_data[2])
    else:
        WB = (mem_pretraining[2], mem_pretraining[3])
        W = mem_pretraining[2]
        B = mem_pretraining[3]

    gradient_descent = algorithm.Momentum(device, WB, eta = 0.003, gamma = 0.95)
    NN_MLP = mlp.MLP(device, mlp_data, gradient_descent, drop_out = drop_out, W = W, B = B)
    return NN_MLP