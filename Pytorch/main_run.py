import time

import torch
import algorithm
import data_info
import display
import MLP
torch.manual_seed(2)




device = algorithm.open_cpu()
start_time = algorithm.open_clock()

train_data, train_label, test_data, test_label = data_info.get_MNIST(device, "/content/AI-project/Pytorch/assets/MNIST/")
algorithm.close_clock_and_show_time(device, start_time)

data = MLP.extract(device, train_data, train_label, [128, 64], list_func = [
    algorithm.ReLU, algorithm.grad_ReLU,
    algorithm.softmax, algorithm.cost
])
gradient_descent = MLP.Momentum(device, data)
neural_network = MLP.Neural_Network(device, data, gradient_descent)

it = neural_network.fit(device, batch_size = 512, max_it = 150)
pred = neural_network.predict(test_data)

algorithm.close_clock_and_show_time(device, start_time)
display.show_accuracy_rate_and_number_iterations(pred, test_label, it)
