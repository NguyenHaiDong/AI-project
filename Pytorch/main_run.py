import time

import torch
import algorithm
import data_info
import display
import MLP
torch.manual_seed(2)




device = algorithm.open_gpu()
print("Tên GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Vẫn là CPU!")
start_time = algorithm.open_clock()

train_data, train_label, test_data, test_label = data_info.get_images(
    device,
    [
        "/content/AI-project/Pytorch/assets/Faces/man",
        "/content/AI-project/Pytorch/assets/Faces/woman",
    ],
    number_images = [9400, 9400],
    width = 64,
    height = 64
)
print(train_label.shape[0], test_label.shape[0])
algorithm.close_clock_and_show_time(device, start_time, "Tổng thời gian đọc dữ liệu")

data = MLP.extract(device, train_data, train_label, [128, 64, 64, 64], list_func = [
    algorithm.ReLU, algorithm.grad_ReLU,
    algorithm.softmax, algorithm.cost
])
gradient_descent = MLP.Momentum(device, data)
neural_network = MLP.Neural_Network(device, data, gradient_descent)

it = neural_network.fit(device, batch_size = 512, max_it = 200)
pred = neural_network.predict(test_data)

algorithm.close_clock_and_show_time(device, start_time)
display.show_accuracy_rate_and_number_iterations(pred, test_label, it)
