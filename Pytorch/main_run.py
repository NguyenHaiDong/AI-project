import time

import torch
import algorithm
import data_info
import display
import MLPs as mlp
import CNNs as cnn
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
    is_flatten = False
)
# print(train_label.shape[0], test_label.shape[0])˝
algorithm.close_clock_and_show_time(device, start_time, "Tổng thời gian đọc dữ liệu")
print(train_data.shape)

data = cnn.extract_for_CNN(
    device, train_data, train_label, 
    layer_size_for_mlp = [128, 64],
    number_conv_layer = 3, C = [16, 32, 64], kernel = [5, 5, 5], s = 1, p = 1,
    max_pooling_kernel_size = 2, max_pooling_stride = 2
)
gradient_descent = algorithm.Adam(device, (data[2], data[3]), eta = 0.0003)
neural_network = cnn.CNN(device, data, gradient_descent, drop_out = 0.08)

it = neural_network.fit(
    device, batch_size = 256, delta = 1e-4, max_it = 100
)
pred = neural_network.predict(test_data)

algorithm.close_clock_and_show_time(device, start_time)
display.show_accuracy_rate_and_number_iterations(pred, test_label, it)