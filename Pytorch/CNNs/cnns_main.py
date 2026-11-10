import algorithm
import torch
import torch.nn.functional as F
import MLPs as mlp
import display
import gc
import handle_image as handle_image
import data_info
import CNNs.init_data as init_data




class CNN:
    def __init__(self, device, data, gradient_descent, drop_out = 0.1):
        (
            self.n, self.Y, self.W, self.B, self.grad_W, self.grad_B, 
            self.A, self.Z, self.dims, self.sz, self.c, self.number_conv_layer, 
            self.s, self.p, self.C, self.kernel, 
            self.final_h, self.final_w, self.layer_size_for_mlp, 
            self.max_pooling_kernel_size, self.max_pooling_stride, self.max_pooling_padding, self.max_pooling_activate, 
            self.has_GAP, self.has_batch_normalization, self.file_pretraining, self.mem_pretraining 
        ) = data

        self.pool_cache = [[] for _ in range(self.number_conv_layer)]
        self.GD = gradient_descent
  
        self.bns, self.count_it, self.step_count = init_data.build_bns(self.mem_pretraining, self.file_pretraining, self.has_batch_normalization, self.number_conv_layer)
        self.NN_MLP = init_data.build_MLP(
            device, self.mem_pretraining, self.file_pretraining, self.layer_size_for_mlp, 
            self.has_GAP, self.final_h, self.final_w, drop_out, self.Y, self.C, self.c, self.n
        )


    def feed_forward(self):
        for i in range(self.number_conv_layer):
            A_flatten = F.unfold(self.A[i], kernel_size = self.kernel[i], stride = self.s, padding = self.p)

            self.Z[i] = self.W[i] @ A_flatten + self.B[i]
            self.Z[i] = self.Z[i].view(self.A[i].shape[0], self.dims[i + 1], self.sz[i][0], self.sz[i][1])

            if self.has_batch_normalization:
                self.Z[i] = self.bns[i].feed_forward(self.Z[i], is_train = True)

            self.A[i + 1] = algorithm.ReLU(self.Z[i])
            if self.max_pooling_activate[i] == 1:
                self.A[i + 1], self.pool_cache[i] = algorithm.max_pooling(
                    self.A[i + 1], self.max_pooling_kernel_size, self.max_pooling_stride, self.max_pooling_padding
                )

        N, C, H, W = self.A[-1].shape
        if self.has_GAP:
            self.NN_MLP.A[0] = self.A[-1].mean(dim=(2, 3))
        else:
            self.NN_MLP.A[0] = self.A[-1].view(N, -1)
        self.NN_MLP.feed_forward()


    def backward_propagation(self, y):
        batch_size = y.shape[0]
        N, C, H, W = self.A[-1].shape
        E = self.NN_MLP.backward_propagation(y)

        if self.has_GAP:
            E = E.view(N, C, 1, 1).expand(N, C, H, W) / (H * W)
        else:
            E = E.view(N, C, H, W)

        for i in range(self.number_conv_layer - 1, -1, -1):
            if self.max_pooling_activate[i] == 1:
                E = algorithm.grad_max_pooling(E, self.pool_cache[i])

            E *= algorithm.grad_ReLU(self.Z[i])
            if self.has_batch_normalization:
                E = self.bns[i].backward_propagation(E)

            E_flatten = E.view(E.shape[0], self.dims[i + 1], -1)
            A_flatten = F.unfold(self.A[i], kernel_size=self.kernel[i], stride=self.s, padding=self.p)

            self.grad_B[i] = torch.sum(E, dim=(0, 2, 3)).unsqueeze(1)
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
        algorithm.clamp_grad(self.grad_W + self.grad_B + self.NN_MLP.grad_W + self.NN_MLP.grad_B, max_norm = 5.0)


    def fit(
        self, device, data, label, patience = 10, batch_size = 64, delta = 1e-4, max_it = 100,
        is_test = False, test_batch = -1, test_data = None, test_label = None,
        path_save_per_test_batch = "None"
    ):
        N = data.shape[0]
        last_cost = 0.0
        patience_count = 0
        batch_size = min(N, batch_size)

        for it in range(1, max_it + 1, 1):
            sample = torch.randperm(N)
            cur_cost = torch.tensor(0.0, device=device)

            for start in range(0, N, batch_size):
                end = min(N, start + batch_size)
                batch_index = sample[start : end]

                self.A[0] = data[batch_index].to(device)
                y = label[batch_index].to(device)

                self.feed_forward()
                self.backward_propagation(y)
                self.GD.gradient_descent(min(1e5, self.step_count), self.W, self.B, self.grad_W, self.grad_B)
                self.NN_MLP.gradient_descent(min(1e5, self.step_count))

                cur_cost += algorithm.cost(y, self.NN_MLP.A[-1]) * (end - start)
                self.step_count += 1

            cur_cost = (cur_cost / N).item()
            if abs(cur_cost - last_cost) <= delta:
                patience_count += 1
                if patience_count > patience:
                    self.count_it += it
                    return it
            else:
                patience_count = 0
            last_cost = cur_cost

            if is_test == True and it % test_batch == 0:
                pred = self.predict(device, test_data)
                self.save(path_save_per_test_batch + " epoch #" + str(int(it / test_batch)) + ".pt")
                display.show_accuracy_rate_and_number_iterations(device, pred, test_label, it)
            print(f"Epoch {it}/{max_it} - Cost: {cur_cost:.4f}")
            self.count_it += 1
        return max_it


    def predict(self, device, out, batch_size = 64):
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

        list_y = []
        with torch.no_grad():
            for start in range(0, out.shape[0], batch_size):
                end = min(out.shape[0], start + batch_size)
                X = out[start : end].to(device)

                for i in range(self.number_conv_layer):
                    X = F.unfold(X, kernel_size = self.kernel[i], stride = self.s, padding = self.p)

                    X = self.W[i] @ X + self.B[i]
                    X = X.view(X.shape[0], self.dims[i + 1], self.sz[i][0], self.sz[i][1])

                    if self.has_batch_normalization:
                        X = self.bns[i].feed_forward(X, is_train = False)

                    X = algorithm.ReLU(X)
                    if self.max_pooling_activate[i]:
                        X = algorithm.max_pooling(X, self.max_pooling_kernel_size, self.max_pooling_stride, self.max_pooling_padding)[0]

                N, C, H, W = X.shape
                if self.has_GAP:
                    X = X.mean(dim=(2, 3))
                else:
                    X = X.view(N, -1)

                y = self.NN_MLP.predict(X)
                list_y.append(y)
        return torch.cat(list_y, dim = 0)


    def save(self, file_name):
        data = []
        data.append(torch.tensor(self.count_it))
        data.append(torch.tensor(self.step_count))

        if self.has_batch_normalization != "None":
            for i in range(len(self.bns)):
                data.append(self.bns[i].var_run)
                data.append(self.bns[i].mu_run)
        data_info.save_to_file([self.W, self.B, self.NN_MLP.W, self.NN_MLP.B], data, file_name)