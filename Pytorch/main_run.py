import time

import torch
import algorithm
import data_info
import display
import MLP
torch.manual_seed(2)




device = algorithm.open_gpu()
start_time = algorithm.open_clock()

# train_data, train_label, test_data, test_label = data_info.get_MNIST(device, "/Users/nguyenhaidong/Desktop/AI/assets/MNIST/")
train_data, train_label, test_data, test_label = data_info.get_images(
    device,
    [
        "/content/AI-project/Pytorch/assets/Faces/man",
        "/content/AI-project/Pytorch/assets/Faces/woman",
    ],
    number_images = [9400, 9400],
    width = 52,
    height = 52
)
# print(train_label.shape[0], test_label.shape[0])
algorithm.close_clock_and_show_time(device, start_time, "Tổng thời gian đọc dữ liệu")

data = MLP.extract(device, train_data, train_label, [128, 64], list_func = [
    algorithm.ReLU, algorithm.grad_ReLU,
    algorithm.softmax, algorithm.cost
])
gradient_descent = MLP.Momentum(device, data, eta = 0.003)
neural_network = MLP.Neural_Network(device, data, gradient_descent)

it = neural_network.fit(device, batch_size = 1024, delta = 5e-2, max_it = 10000)
pred = neural_network.predict(test_data)

algorithm.close_clock_and_show_time(device, start_time)
display.show_accuracy_rate_and_number_iterations(pred, test_label, it)
