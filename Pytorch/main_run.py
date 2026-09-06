import time

import torch
import algorithm
import data_info
import display
import neural_network as nn
torch.manual_seed(2)




device = algorithm.open_gpu()
start_time = algorithm.open_clock()

# train_data, train_label, test_data, test_label = data_info.get_MNIST(device, "/Users/nguyenhaidong/Desktop/AI/assets/MNIST/")
train_data, train_label, test_data, test_label = data_info.get_images(
    device,
    [
        "/Users/nguyenhaidong/Desktop/AI/assets/Faces/man",
        "/Users/nguyenhaidong/Desktop/AI/assets/Faces/woman",
    ],
    number_images = [9400, 9400],
    width = 64,
    height = 64,
    is_flatten = True
)
# print(train_label.shape[0], test_label.shape[0])˝
algorithm.close_clock_and_show_time(device, start_time, "Tổng thời gian đọc dữ liệu")

data = nn.extract_for_MLP(device, train_data, train_label, [128, 64], list_func = [
    algorithm.ReLU, algorithm.grad_ReLU,
    algorithm.softmax, algorithm.cost
])
gradient_descent = nn.Momentum(device, data, eta = 0.003, gamma = 0.99)
neural_network = nn.MLP(device, data, gradient_descent, drop_out = 0.08)

it = neural_network.fit(
    device, batch_size = 1024, delta = 1e-4, max_it = 150,
    # test_batch = 10, test_data = test_data, test_label = test_label
)
pred = neural_network.predict(test_data)

algorithm.close_clock_and_show_time(device, start_time)
display.show_accuracy_rate_and_number_iterations(pred, test_label, it)

print("HI")