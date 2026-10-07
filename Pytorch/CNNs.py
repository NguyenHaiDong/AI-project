import algorithm
import torch
import torch.nn.functional as F
import MLPs as mlp
import display
import gc
import handle_image as handle_image




def extract_for_CNN(
    device, X, Y, layer_size_for_mlp = [100, 100], c = -1, list_func = None,
    number_conv_layer = 3, C = [16, 16, 16], kernel = [5, 5, 5], s = 1, p = 1,
    max_pooling_kernel_size = 3, max_pooling_stride = 3, max_pooling_padding = 0
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

        cur_h = (cur_h + 2 * max_pooling_padding - max_pooling_kernel_size) // max_pooling_stride + 1
        cur_w = (cur_w + 2 * max_pooling_padding - max_pooling_kernel_size) // max_pooling_stride + 1
        h = cur_h
        w = cur_w

    dims = [X.shape[1]] + C
    W = [torch.randn(dims[i + 1], dims[i] * kernel[i] * kernel[i], device = device) * algorithm.get_better_randn(dims[i] * kernel[i] * kernel[i]) for i in range(number_conv_layer)]
    B = [torch.zeros(dims[i + 1], 1, device = device) for i in range(number_conv_layer)]

    grad_W = [torch.zeros_like(w) for w in W]
    grad_B = [torch.zeros_like(b) for b in B]

    A = [None for _ in range(number_conv_layer + 1)]
    Z = [None for _ in range(number_conv_layer)]

    return (
        n, Y, W, B, grad_W, grad_B, A, Z, dims, sz, c, number_conv_layer, s, p, C, kernel, h, w, layer_size_for_mlp, 
        max_pooling_kernel_size, max_pooling_stride, max_pooling_padding
    )



class CNN:
    def __init__(self, device, data, gradient_descent, drop_out = 0.1):
        self.n = data[0]
        self.Y = data[1]
        self.W = data[2]
        self.B = data[3]
        self.grad_W = data[4]
        self.grad_B = data[5]

        self.A = data[6]
        self.Z = data[7]

        self.dims = data[8]
        self.sz = data[9]
        self.c = data[10]
        self.number_conv_layer = data[11]

        self.s = data[12]
        self.p = data[13]
        self.C = data[14]
        self.kernel = data[15]

        self.final_h = data[16]
        self.final_w = data[17]
        self.layer_size_for_mlp = data[18]

        self.max_pooling_kernel_size = data[19]
        self.max_pooling_stride = data[20]
        self.max_pooling_padding = data[21]

        self.pool_cache = [[] for _ in range(self.number_conv_layer)]
        self.GD = gradient_descent

        mlp_data = mlp.extract_for_MLP(
            device, torch.zeros(self.n, self.C[-1] * self.final_h * self.final_w , device = device), self.Y, 
            number_neurons_per_layer = self.layer_size_for_mlp, 
            list_func = [
                algorithm.ReLU, algorithm.grad_ReLU,
                algorithm.softmax, algorithm.cost
            ],
            is_need_to_convert_one_hot_coding = False
        )
        gradient_descent = algorithm.Momentum(device, (mlp_data[1], mlp_data[2]), eta = 0.003, gamma = 0.95)
        self.NN_MLP = mlp.MLP(device, mlp_data, gradient_descent, drop_out = drop_out)


    def feed_forward(self):
        for i in range(self.number_conv_layer):
            A_flatten = F.unfold(self.A[i], kernel_size = self.kernel[i], stride = self.s, padding = self.p)

            self.Z[i] = self.W[i] @ A_flatten + self.B[i]
            self.Z[i] = self.Z[i].view(self.A[i].shape[0], self.dims[i + 1], self.sz[i][0], self.sz[i][1])

            self.A[i + 1] = algorithm.ReLU(self.Z[i])
            self.A[i + 1], self.pool_cache[i] = algorithm.max_pooling(
                self.A[i + 1], self.max_pooling_kernel_size, self.max_pooling_stride, self.max_pooling_padding
            )

        N, C, H, W = self.A[-1].shape
        self.NN_MLP.A[0] = self.A[-1].view(N, C * H * W)
        self.NN_MLP.feed_forward()


    def backward_propagation(self, y):
        N, C, H, W = self.A[-1].shape
        E = self.NN_MLP.backward_propagation(y)
        E = E.view(N, C, H, W)

        for i in range(self.number_conv_layer - 1, -1, -1):
            E = algorithm.grad_max_pooling(E, self.pool_cache[i])
            E *= algorithm.grad_ReLU(self.Z[i])

            self.grad_B[i] = torch.sum(E, dim=(0, 2, 3)).unsqueeze(1)

            E_flatten = E.view(E.shape[0], self.dims[i + 1], -1)
            A_flatten = F.unfold(self.A[i], kernel_size=self.kernel[i], stride=self.s, padding=self.p)
            self.grad_W[i] = torch.bmm(E_flatten, A_flatten.transpose(1, 2)).sum(dim=0)

            if i > 0:
                E_unfold = self.W[i].t() @ E_flatten

                E = F.fold(
                    E_unfold,
                    output_size=self.A[i].shape[2:],
                    kernel_size=self.kernel[i],
                    stride=self.s,
                    padding=self.p
                )


    def fit(
        self, device, data, label, patience = 10, batch_size = 64, delta = 1e-4, max_it = 100,
        is_test = False, test_batch = -1, test_data = None, test_label = None
    ):
        N = data.shape[0]
        last_cost = 0.0
        patience_count = 0
        batch_size = min(N, batch_size)

        for it in range(1, max_it + 1, 1):
            sample = torch.randperm(N, device=device)
            X = data[sample]
            Y = label[sample]
            cur_cost = torch.tensor(0.0, device=device)

            for start in range(0, N, batch_size):
                end = min(N, start + batch_size)

                self.A[0] = X[start : end]
                y = Y[start : end]

                self.feed_forward()
                self.backward_propagation(y)
                self.GD.gradient_descent(it, self.W, self.B, self.grad_W, self.grad_B)
                self.NN_MLP.gradient_descent(it)

                cur_cost += algorithm.cost(y, self.NN_MLP.A[-1]) * (end - start)

            cur_cost = (cur_cost / N).item()
            if abs(cur_cost - last_cost) <= delta:
                patience_count += 1
                if patience_count > patience:
                    return it
            else:
                patience_count = 0
            last_cost = cur_cost

            if is_test == True and it % test_batch == 0:
                pred = self.predict(test_data)
                display.show_accuracy_rate_and_number_iterations(pred, test_label, it)
            print(f"Epoch {it}/{max_it} - Cost: {cur_cost:.4f}")
        return max_it


    def predict(self, data):
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

        out = data
        for i in range(self.number_conv_layer):
            out = F.unfold(out, kernel_size = self.kernel[i], stride = self.s, padding = self.p)

            out = self.W[i] @ out + self.B[i]
            out = out.view(out.shape[0], self.dims[i + 1], self.sz[i][0], self.sz[i][1])

            out = algorithm.ReLU(out)
            out = algorithm.max_pooling(out, self.max_pooling_kernel_size, self.max_pooling_stride, self.max_pooling_padding)[0]

        N, C, H, W = out.shape
        out = out.view(N, C * W * H)
        return self.NN_MLP.predict(out)