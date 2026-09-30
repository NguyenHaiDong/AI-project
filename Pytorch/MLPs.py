import torch
import algorithm
import display
import torch.nn.functional as F




def extract_for_MLP(
    device, X, Y, number_neurons_per_layer = [100, 100], c = -1, list_func = None
):
    n = X.shape[0]
    d = X.shape[1]
    if c == -1:
        c = torch.unique(Y).numel()

    dims = [d] + number_neurons_per_layer + [c]
    number_layers = len(dims) - 1

    if len(list_func) / 2 < number_layers:
        delta = int(number_layers - len(list_func) / 2)
        list_func = [list_func[-4], list_func[-3]] * delta + list_func

    W = [torch.randn(dims[i], dims[i + 1], device=device) * algorithm.get_better_randn(dims[i]) for i in range(number_layers)]
    B = [torch.zeros(1, dims[i + 1], device=device) for i in range(number_layers)]

    return (X, Y, W, B, n, d, c, number_layers, list_func)




class MLP:
    def __init__(self, device, data, GD, drop_out = 0.2, is_need_to_convert_Y = True):
        self.X = data[0]
        if is_need_to_convert_Y:
            self.Y = algorithm.convert_to_one_hot_coding(device, data[1], data[6])

        self.W = data[2]
        self.B = data[3]

        self.n = data[4]
        self.d = data[5]
        self.c = data[6]
        self.number_layers = data[7]
        self.list_func = data[8]
        self.drop_out = drop_out
        self.scale = 1.0 / (1.0 - self.drop_out)

        self.grad_W = [torch.zeros_like(w, device=device) for w in self.W]
        self.grad_B = [torch.zeros_like(b, device=device) for b in self.B]

        self.A = [self.X] + [None for _ in range(self.number_layers)]
        self.Z = [None for _ in range(self.number_layers)]
        self.GD = GD


    def feed_forward(self):
        for i in range(self.number_layers):
            rand = (torch.rand_like(self.A[i]) > self.drop_out).float() * self.scale
            A_filter = torch.mul(self.A[i], rand)

            self.Z[i] = torch.addmm(self.B[i], A_filter, self.W[i])
            self.A[i + 1] = self.list_func[2 * i](self.Z[i])


    def backward_propagation(self, y):
        E = (self.A[-1] - y) / y.shape[0]

        for i in range(self.number_layers - 1, -1, -1):
            self.grad_W[i] = torch.mm(self.A[i].T, E)
            self.grad_B[i] = torch.sum(E, dim=0, keepdim=True)

            E = torch.mm(E, self.W[i].T)
            if i > 0:
                E.mul_(self.list_func[2 * (i - 1) + 1](self.Z[i - 1]))
        return E


    def gradient_descent(self, it):
        self.GD.gradient_descent(it, self.W, self.B, self.grad_W, self.grad_B)


    def fit(
        self, device, data = -1, label = -1, patience = 10, batch_size = 64, delta = 1e-4, max_it = 100, 
        is_test = False, test_batch = -1, test_data = None, test_label = None
    ):
        N = data.shape[0]
        last_cost = 0.0
        patience_count = 0
        batch_size = min(N, batch_size)

        if data == -1:
            data = self.X
        if label == -1:
            label = self.Y

        for it in range(1, max_it + 1):
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

                cur_cost += self.list_func[-1](y, self.A[-1]) * (end - start)

            cur_cost = (cur_cost / N).item()
            if abs(cur_cost - last_cost) <= delta:
                patience_count += 1
                if patience_count > patience:
                    return it
            else:
                patience_count = 0
            last_cost = cur_cost

            if is_test == True and it % test_batch == 0:
                print(f"Done It #{it}")
            elif it % test_batch == 0:
                pred = self.predict(test_data)
                display.show_accuracy_rate_and_number_iterations(pred, test_label, it)
        return max_it


    def predict(self, data):
        out = data
        for i in range(self.number_layers):
            out = torch.addmm(self.B[i], out, self.W[i])
            out = self.list_func[2 * i](out)
        return torch.argmax(out, dim=1).flatten()