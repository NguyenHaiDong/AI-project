import os
# os.environ["VECLIB_MAXIMUM_THREADS"] = "5"
# os.environ["OPENBLAS_NUM_THREADS"] = "5"
# os.environ["NUMEXPR_NUM_THREADS"] = "5"
import time
import gc

import torch
import algorithm
import data_info
import display
import CNNs.cnns as cnn
import handle_image as handle_image

torch.manual_seed(2)

# Tắt vĩnh viễn chế độ dựng đồ thị đạo hàm
torch.set_grad_enabled(False)
gc.collect()

# # 2. Giải phóng toàn bộ bộ nhớ đệm mà PyTorch đang giữ nhưng chưa dùng
# torch.cuda.empty_cache()
# torch.cuda.ipc_collect()

attributes_EN = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
]

attributes_VN = [
    "Râu lún phún", "Lông mày cánh cung", "Thu hút", "Bọng mắt", "Hói đầu",
    "Tóc mái", "Môi dày", "Mũi to", "Tóc đen", "Tóc vàng",
    "Ảnh mờ", "Tóc nâu", "Lông mày rậm", "Mũm mĩm", "Nọng cằm",
    "Đeo kính", "Râu cằm", "Tóc bạc", "Trang điểm đậm", "Gò má cao",
    "Nam giới", "Miệng hé mở", "Ria mép", "Mắt hẹp", "Không có râu",
    "Mặt trái xoan", "Da nhợt nhạt", "Mũi nhọn", "Đường chân tóc lùi", "Má hồng",
    "Tóc mai", "Cười", "Tóc thẳng", "Tóc lượn sóng", "Đeo khuyên tai",
    "Đội mũ", "Đánh son", "Đeo vòng cổ", "Đeo cà vạt", "Trẻ tuổi"
]


device = algorithm.open_gpu()
start_time = algorithm.open_clock()

# train_data, train_label, test_data, test_label = data_info.get_images(
#     device,
#     folder_paths = [
#         "/Users/nguyenhaidong/DES/AI/assets/Real Data/Man"
#     ],
#     number_images = 1,
#     channels = 3,
#     width = 96,
#     height = 96,
#     divide = [0.0, 100.0],
#     is_shuffle = False,
#     is_flatten = False
# )

train_data, train_label, test_data, test_label = data_info.load_celeba(
    image_path = "/kaggle/working/AI-project/Pytorch/assets/LARGE_FACE/img_align_celeba",
    attribute_path = "/kaggle/working/AI-project/Pytorch/assets/LARGE_FACE/list_attr_celeba.csv",
    number_images = 50000,
    channels = 3,
    labels = None,  
    image_size = [96, 96],
    divide = [75.0, 25.0],
)

print(train_data.shape)
algorithm.close_clock_and_show_time(device, start_time, "Tổng thời gian đọc dữ liệu")

data = cnn.extract_for_CNN(
    device, train_data, train_label, 
    layer_size_for_mlp = [1024, 256, 64],
    number_conv_layer = 7, 
    C = [16, 32, 64, 128, 256, 512, 1024],
    kernel = [3, 3, 3, 3, 3, 3, 3], s = 1, p = 1,
    max_pooling_kernel_size = 2, max_pooling_stride = 2,
    max_pooling_activate = [1, 0, 1, 0, 1, 0, 1], 
    has_GAP = True,
    has_batch_normalization = True,
    c = 40
)
gradient_descent = algorithm.Adam(device, (data[2], data[3]), eta = 0.005, weight_decay = 0.1, decay_eta = 0.03)
neural_network = cnn.CNN(device, data, gradient_descent, drop_out = 0.1)
# neural_network.count_it = 0

it = neural_network.fit(
    device, train_data, data[1], batch_size = 128, delta = 1e-4, max_it = 100,
    is_test = True, test_batch = 5, test_data = test_data, test_label = test_label,
    path_save_per_test_batch = "mem"
)
# it = 0
pred = neural_network.predict(device, test_data)
neural_network.save("mem_attri.pt")

# display.show_image_and_attribute(test_data[0], width = 96, height = 96, label = pred[0], attributes = attributes_VN)

algorithm.close_clock_and_show_time(device, start_time)
display.show_accuracy_rate_and_number_iterations(device, pred, test_label, it)
