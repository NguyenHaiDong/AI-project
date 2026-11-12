import torch




class Adam:
    def __init__(self, device, data, beta1 = 0.9, beta2 = 0.999, eta = 3e-4, epsilon = 1e-7, weight_decay = 0.01, decay_eta = 1e-4):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        self.eta = eta
        self.decay_eta = decay_eta

        self.number_layers = len(data[0])
        self.b1 = 1.0
        self.b2 = 1.0

        self.W_M = [torch.zeros_like(w, device=device) for w in data[0]]
        self.W_V = [torch.zeros_like(w, device=device) for w in data[0]]

        self.B_M = [torch.zeros_like(b, device=device) for b in data[1]]
        self.B_V = [torch.zeros_like(b, device=device) for b in data[1]]


    def gradient_descent(self, it, W, B, grad_W, grad_B):
        cur_eta = self.eta / (1.0 + self.decay_eta * it)
        self.b1 *= self.beta1
        self.b2 *= self.beta2

        b1 = 1.0 - self.b1
        b2 = 1.0 - self.b2
        a = cur_eta * (b2 ** 0.5) / b1
        b = self.epsilon * (b2 ** 0.5)
        decay_factor = 1.0 - cur_eta * self.weight_decay

        for i in range(self.number_layers):
            self.W_M[i].mul_(self.beta1).add_(grad_W[i], alpha = (1.0 - self.beta1))
            self.W_V[i].mul_(self.beta2).addcmul_(grad_W[i], grad_W[i], value = (1.0 - self.beta2))

            self.B_M[i].mul_(self.beta1).add_(grad_B[i], alpha = (1.0 - self.beta1))
            self.B_V[i].mul_(self.beta2).addcmul_(grad_B[i], grad_B[i], value = (1.0 - self.beta2))

            W[i].mul_(decay_factor)

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