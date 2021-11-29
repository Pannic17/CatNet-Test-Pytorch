import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model_v2 import MobileNetV2
import detection as detection
import cv2


def main(image):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    # file_name = input('Filename:')
    # img_path = "H:/Project/21ACB/Test6_mobilenet/test/" + file_name
    # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # img = Image.open(img_path)
    # plt.imshow(img)
    # [N, C, H, W]
    image = data_transform(image)
    # expand batch dimension
    image = torch.unsqueeze(image, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = MobileNetV2(num_classes=7).to(device)
    # load model weights
    model_weight_path = "./CatMobileNet.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(image.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "Class: {}   Probability: {:.3}".format(class_indict[str(predict_cla)],
                                                        predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    # return
    # print(class_indict[str(predict_cla)])
    plt.show()


if __name__ == '__main__':
    file_name = input("Filename: ")
    img_path = "H:/Project/21ACB/Test6_mobilenet/test/" + file_name
    assert os.path.exists(img_path), "File: '{}' dose not exist.".format(img_path)
    # get cat face and coordinate
    cat, coordinate = detection.cut(img_path)
    if cat is not None:
        # modify the image via opencv
        cat = detection.recolor(cat)
        full_cat = cv2.imread(img_path)
        full_cat = cv2.rectangle(full_cat,
                                 (coordinate[0], coordinate[1]),
                                 (coordinate[2], coordinate[3]),
                                 (0, 0, 255), 2)
        # transform the image to PIL image type
        cat = Image.fromarray(cat)
        full_cat = detection.recolor(full_cat)
        # show and test
        plt.imshow(full_cat)
        print("Coordinates:",
              coordinate[0], coordinate[1], coordinate[2], coordinate[3])
        print("Size:", coordinate[4], "*", coordinate[4])
        main(cat)
    else:
        print("Cannot detect cat!!!")
