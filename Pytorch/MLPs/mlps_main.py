import torch
import Algorithm.algorithm as algorithm
import torch.nn.functional as F
import data_info




class MLP:
    def __init__(self, device, data, GD, drop_out = 0.2):
        (
            self.W, self.B, self.A, self.Z, self.grad_B, self.grad_W, 
            self.n, self.d, self.c, self.number_layers, self.file_pretraining
        ) = data

        self.list_func_forward  = [algorithm.ReLU for _ in range(self.number_layers - 1)] + [algorithm.sigmoid]
        self.list_func_backward = [algorithm.grad_sigmoid] + [algorithm.grad_ReLU for _ in range(self.number_layers - 1)]

        self.drop_out = algorithm.Drop_out(drop_out)
        self.fit_manager = algorithm.Fit_Manager(device)
        self.GD = GD


    def feed_forward(self):
        self.A, self.Z = algorithm.feed_forward(
            self.number_layers, self.A, self.Z, self.W, self.B, 
            [self.drop_out.run],
            self.list_func_forward
        )

    def backward_propagation(self, y):
        E, self.grad_W, self.grad_B = algorithm.backward_propagation(
            self.number_layers, y, self.A, self.Z, self.W, self.grad_W, self.grad_B,
            self.list_func_backward
        )
        return E

    def gradient_descent(self, it):
        self.GD.gradient_descent(it, self.W, self.B, self.grad_W, self.grad_B)


    def fit(
        self, device, data, label, patience = 10, batch_size = 64, delta = 1e-4, max_it = 100, 
        is_test = False, test_batch = -1, test_data = None, test_label = None
    ):
        N = data.shape[0]
        batch_size = min(N, batch_size)

        for it in range(1, max_it + 1):
            sample = torch.randperm(N, device=device)

            for start in range(0, N, batch_size):
                end = min(N, start + batch_size)
                batch_index = sample[start : end]

                self.A[0] = data[batch_index].to(device)
                y = label[batch_index].to(device)

                self.feed_forward()
                self.backward_propagation(y)
                self.GD.gradient_descent(it, self.W, self.B, self.grad_W, self.grad_B)
                self.fit_manager.update_cost(y, self.A[-1], self.list_func_forward[-1], start, end)

            if self.fit_manager.run(N, delta, patience, it, is_test, test_batch, test_data, test_label):
                return it
        return max_it


    def predict(self, data, threshold = 0.5):
        return algorithm.predict(self.number_layers, data, self.W, self.B, self.list_func_forward, threshold)

    def save(self, file_name):
        data_info.save_to_file([self.W, self.B], file_name)

    def load(self, file_name):
        self.W, self.B = data_info.load_from_file(file_name)