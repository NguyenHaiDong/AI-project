import torch
import algorithm




def extract(device, X, Y, number_neurons_per_layer = [100, 100], c = -1, list_func = None):
    n = X.shape[0]
    d = X.shape[1]
    if c == -1:
        c = len(torch.unique(Y))

    dims = [d] + number_neurons_per_layer + [c]
    number_layers = len(dims) - 1

    if len(list_func) / 2 < number_layers:
        delta = int(number_layers - len(list_func) / 2)
        list_func = [list_func[-4], list_func[-3]] * delta + list_func

    W = [torch.randn(dims[i], dims[i + 1], device=device) * ((2.0 / dims[i]) ** 0.5) for i in range(number_layers)]
    B = [torch.zeros(1, dims[i + 1], device=device) for i in range(number_layers)]

    return (W, B, X, Y, n, d, c, number_layers, list_func)




class Adam:
    def __init__(self, device, data, beta1 = 0.9, beta2 = 0.999, eta = 3e-4, epsilon = 1e-7, weight_decay = 0.01):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        self.number_layers = len(data[0])
        self.b1 = 1.0
        self.b2 = 1.0

        self.W_M = [torch.zeros_like(w, device=device) for w in data[0]]
        self.W_V = [torch.zeros_like(w, device=device) for w in data[0]]

        self.B_M = [torch.zeros_like(b, device=device) for b in data[1]]
        self.B_V = [torch.zeros_like(b, device=device) for b in data[1]]


    def gradient_descent(self, it, W, B, grad_W, grad_B):
        self.b1 *= self.beta1
        self.b2 *= self.beta2

        b1 = 1.0 - self.b1
        b2 = 1.0 - self.b2
        a = self.eta * (b2 ** 0.5) / b1
        b = self.epsilon * (b2 ** 0.5)
        decay_factor = 1.0 - self.eta * self.weight_decay

        for i in range(self.number_layers):
            self.W_M[i].mul_(self.beta1).add_(grad_W[i], alpha = (1.0 - self.beta1))
            self.W_V[i].mul_(self.beta2).addcmul_(grad_W[i], grad_W[i], value = (1.0 - self.beta2))

            self.B_M[i].mul_(self.beta1).add_(grad_B[i], alpha = (1.0 - self.beta1))
            self.B_V[i].mul_(self.beta2).addcmul_(grad_B[i], grad_B[i], value = (1.0 - self.beta2))

            # W[i].mul_(decay_factor)

            denom_W = torch.sqrt(self.W_V[i]).add_(b)
            W[i].addcdiv_(self.W_M[i], denom_W, value = -a)

            denom_B = torch.sqrt(self.B_V[i]).add_(b)
            B[i].addcdiv_(self.B_M[i], denom_B, value = -a)




class Momentum:
    def __init__(self, device, data, eta = 0.01, gamma = 0.9, decay_rate = 1e-3, weight_decay = 0.01):
        self.eta = eta
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.weight_decay = weight_decay

        self.number_layers = len(data[0])
        self.V_W = [torch.zeros_like(w, device=device) for w in data[0]]
        self.V_B = [torch.zeros_like(b, device=device) for b in data[1]]


    def gradient_descent(self, it, W, B, grad_W, grad_B):
        cur_eta = self.eta / (1.0 + self.decay_rate * it)
        decay_factor = 1.0 - cur_eta * self.weight_decay
        eta_gamma = cur_eta * self.gamma

        for i in range(self.number_layers):
            self.V_W[i].mul_(self.gamma).add_(grad_W[i])
            self.V_B[i].mul_(self.gamma).add_(grad_B[i])

            W[i].mul_(decay_factor)
            W[i].add_(grad_W[i], alpha = -cur_eta).add_(self.V_W[i], alpha = -eta_gamma)
            B[i].add_(grad_B[i], alpha = -cur_eta).add_(self.V_B[i], alpha = -eta_gamma)




class Neural_Network:
    def __init__(self, device, data, GD):
        self.W = data[0]
        self.B = data[1]
        self.X = data[2]
        self.Y = algorithm.convert_to_one_hot_coding(device, data[3], data[6])

        self.n = data[4]
        self.d = data[5]
        self.c = data[6]
        self.number_layers = data[7]
        self.list_func = data[8]
        
        self.grad_W = [torch.zeros_like(w, device=device) for w in self.W]
        self.grad_B = [torch.zeros_like(b, device=device) for b in self.B]

        self.A = [self.X] + [None for _ in range(self.number_layers)]
        self.Z = [None for _ in range(self.number_layers)]
        self.GD = GD


    def feed_forward(self):
        for i in range(self.number_layers):
            self.Z[i] = torch.addmm(self.B[i], self.A[i], self.W[i])
            self.A[i + 1] = self.list_func[2 * i](self.Z[i])


    def backward_propagation(self, y):
        E = (self.A[-1] - y) / y.shape[0]

        for i in range(self.number_layers - 1, -1, -1):
            self.grad_W[i] = torch.mm(self.A[i].T, E)
            self.grad_B[i] = torch.sum(E, dim=0, keepdim=True)
            if i > 0:
                E = torch.mm(E, self.W[i].T)
                E.mul_(self.list_func[2 * (i - 1) + 1](self.Z[i - 1]))


    def fit(self, device, patience = 10, batch_size = 64, delta = 1e-4, max_it = 100):
        last_cost = 0.0
        patience_count = 0
        batch_size = min(self.n, batch_size)

        for it in range(1, max_it + 1):
            sample = torch.randperm(self.n, device=device)
            X = self.X[sample]
            Y = self.Y[sample]
            cur_cost = torch.tensor(0.0, device=device)

            for start in range(0, self.n, batch_size):
                end = min(self.n, start + batch_size)

                self.A[0] = X[start : end]
                y = Y[start : end]

                self.feed_forward()
                self.backward_propagation(y)
                self.GD.gradient_descent(it, self.W, self.B, self.grad_W, self.grad_B)

                cur_cost += self.list_func[-1](y, self.A[-1]) * (end - start)

            cur_cost = (cur_cost / self.n).item()
            if abs(cur_cost - last_cost) <= delta:
                patience_count += 1
                if patience_count > patience:
                    return it
            else:
                patience_count = 0
            last_cost = cur_cost

            print(f"Done It #{it}")
        return max_it


    def predict(self, data):
        out = data
        for i in range(self.number_layers):
            out = torch.addmm(self.B[i], out, self.W[i])
            out = self.list_func[2 * i](out)
        return torch.argmax(out, dim=1).flatten()
