import torch
import display




class Batch_Normalization:
    def __init__(self, epsilon = 1e-5, momentum = 0.2, var_run = torch.tensor(1.0), mu_run = torch.tensor(1.0)):
        self.epsilon = epsilon
        self.momentum = momentum

        self.mu_run = mu_run
        self.var_run = var_run
        self.inv_var_run = torch.sqrt(var_run + self.epsilon)
        self.inv_var = 1.0
        self.Z_hat = None

        self.gamma = 1.0
        self.beta = 0.0


    def feed_forward(self, Z, is_train = True):
        N, C, H, W = Z.shape
        m = N * H * W

        if is_train:
            mu = torch.sum(Z, dim = (0, 2, 3), keepdim = True) / m
            var = torch.var(Z, dim = (0, 2, 3), keepdim = True, unbiased=False)
            self.inv_var = torch.sqrt(var + self.epsilon)
            
            self.Z_hat = (Z - mu) / self.inv_var
            Z_out = self.Z_hat * self.gamma + self.beta

            self.mu_run = self.mu_run * (1 - self.momentum) + mu.detach() * self.momentum
            self.var_run = self.var_run * (1 - self.momentum) + var.detach() * self.momentum
            self.inv_var_run = torch.sqrt(self.var_run + self.epsilon)
        else:
            self.Z_hat = (Z - self.mu_run) / self.inv_var_run
            Z_out = self.Z_hat * self.gamma + self.beta
        return Z_out


    def backward_propagation(self, Z_hat_grad, eta = 0.01):
        N, C, H, W = Z_hat_grad.shape
        m = N * H * W
        grad_beta = torch.mean(Z_hat_grad, dim = (0, 2, 3), keepdim = True)
        grad_gamma = torch.mean(Z_hat_grad * self.Z_hat, dim = (0, 2, 3), keepdim = True)

        Z_grad = (self.gamma / self.inv_var) * (Z_hat_grad - grad_beta - self.Z_hat * grad_gamma)

        self.gamma -= eta * grad_gamma
        self.beta -= eta * grad_beta
        return Z_grad




class Drop_out:
    def __init__(self, drop_out):
        self.drop_out = drop_out
        self.scale = 1.0 / (1.0 - self.drop_out)

    def run(self, A):
        rand = (torch.rand_like(A) > self.drop_out).float() * self.scale
        A_filter = torch.mul(A, rand)
        return A_filter


class Fit_Manager:
    def __init__(self, device):
        self.cur_cost = torch.tensor(0.0, device=device)
        self.last_cost = 0.0
        self.patience_count = 0
        pass

    def run(self, N, delta, patience, it, is_test = False, test_batch = 10, test_data = None, test_label = None):
        self.cur_cost = (self.cur_cost / N).item()
        if abs(self.cur_cost - self.last_cost) <= delta:
            self.patience_count += 1
            if self.patience_count > patience:
                return True
        else:
            self.patience_count = 0
        self.last_cost = self.cur_cost

        if is_test == True and it % test_batch == 0:
            print(f"Done It #{it}")
        elif it % test_batch == 0:
            pred = self.predict(test_data)
            display.show_accuracy_rate_and_number_iterations(pred, test_label, it)
        return False

    def update_cost(self, y, A, func, start, end):
        self.cur_cost += func(y, A) * (end - start)




def feed_forward(number_layers, A, Z, W, B, list_func1, list_func2):
    if len(list_func1) == 0:
        for i in range(number_layers):
            Z[i] = torch.addmm(B[i], A[i], W[i])
            A[i + 1] = list_func2[i](Z[i])
        return A, Z

    for i in range(number_layers):
        A_cur = A[i]
        if i > 0:
            for j in range(0, len(list_func1)):
                A_cur = list_func1[j](A_cur)

        Z[i] = torch.addmm(B[i], A_cur, W[i])
        A[i + 1] = list_func2[i](Z[i])
    return A, Z


def backward_propagation(number_layers, y, A, Z, W, grad_W, grad_B, list_func):
    E = (A[-1] - y) / (y.shape[0] * y.shape[1])

    for i in range(number_layers - 1, -1, -1):
        grad_W[i] = torch.mm(A[i].T, E)
        grad_B[i] = torch.sum(E, dim=0, keepdim=True)

        E = torch.mm(E, W[i].T)
        if i > 0:
            E.mul_(list_func[i](Z[i - 1]))
    return E, grad_W, grad_B


def predict(number_layers, data, W, B, list_func, threshold = 0.5):
    out = data
    for i in range(number_layers):
        out = torch.addmm(B[i], out, W[i])
        out = list_func[i](out)
    return (out >= threshold).float()