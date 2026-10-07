import os
# os.environ["VECLIB_MAXIMUM_THREADS"] = "5"
# os.environ["OPENBLAS_NUM_THREADS"] = "5"
# os.environ["NUMEXPR_NUM_THREADS"] = "5"
import time

import torch
import algorithm
import data_info
import display
import MLPs as mlp
import CNNs as cnn
import handle_image as handle_image

torch.manual_seed(2)




device = algorithm.open_gpu()
start_time = algorithm.open_clock()

# train_data, train_label, test_data, test_label = data_info.get_MNIST(device, "/Users/nguyenhaidong/Desktop/AI/assets/MNIST/")
train_data, train_label, test_data, test_label = data_info.get_images(
    device,
    [
        "/Users/nguyenhaidong/Desktop/AI/assets/Faces/man",
        "/Users/nguyenhaidong/Desktop/AI/assets/Faces/woman"
    ],
    number_images = [1000, 1000],
    width = 64,
    height = 64,
    is_flatten = False
)
# handle_image.convert_color(train_data, [1.5])
# train_data, train_label = handle_image.rotate_data(train_data, train_label)

# print(train_label.shape[0], test_label.shape[0])˝
algorithm.close_clock_and_show_time(device, start_time, "Tổng thời gian đọc dữ liệu")
print(train_data.shape)

data = cnn.extract_for_CNN(
    device, train_data, train_label, 
    layer_size_for_mlp = [64, 32],
    number_conv_layer = 3, 
    C = [16, 32, 32],
    kernel = [3, 3, 3], s = 1, p = 1,
    max_pooling_kernel_size = 2, max_pooling_stride = 2
)
gradient_descent = algorithm.Adam(device, (data[2], data[3]), eta = 0.0002)
neural_network = cnn.CNN(device, data, gradient_descent, drop_out = 0.08)

# angles = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
# for i in range(len(angles)):
#     train_data = handle_image.rotate_image(train_data, angle = angles[i])
#     it = neural_network.fit(
#         device, batch_size = 128, delta = 1e-4, max_it = 50,
#         is_test = True, test_batch = 10, test_data = test_data, test_label = test_label 
#     )
# pred = neural_network.predict(test_data)

it = neural_network.fit(
    device, batch_size = 128, delta = 1e-4, max_it = 100,
    is_test = True, test_batch = 10, test_data = test_data, test_label = test_label 
)
pred = neural_network.predict(test_data)

algorithm.close_clock_and_show_time(device, start_time)
display.show_accuracy_rate_and_number_iterations(pred, test_label, it)



# print(train_data[0])
# train_data[0] = handle_image.rotate_image(train_data[0], angle = 15)
# print(train_data[0])
# display.show_image(train_data[0], height = 128, width = 128)