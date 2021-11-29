import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model_v2 import MobileNetV2


def main():

    # %% SUPER_PARAMETER DEFINITION

    # select train device： GPU if CUDA is available, CPU if CUDA is not available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("using {} device.".format(device))

    # set batch size to 16 (means train 16 images at the same time)
    batch_size = 16
    # set epochs to 5 (mean train 5 times)
    epochs = 24

    # %% PRE_PROCESS DATA

    # split data to train and validation
    data_transform = {
        # resize image of any size to fixed size (224*224)
        # randomly flip image horizontally by ratio of 0.3
        # transform PIL image or ndarray to tensor
        # normalize the picture
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(p=0.3),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # get data root path
    data_root = os.path.abspath(os.path.join(os.getcwd()))
    # flower data set path
    image_path = os.path.join(data_root, "data_set", "cat_data")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # locate train dataset and load images via datasets.ImageFolder
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    # count train images
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4} get the label list
    flower_list = train_dataset.class_to_idx
    # create a dictionary of (val, key)
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)  # indent缩进
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    num_workers = 8  # i7-10700
    print('Using {} dataloader workers every process ///CPU:i7-10700'.format(num_workers))

    # load data to train dataset
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)

    # locate validate dataset and load images via datasets.ImageFolder
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    # count validate images
    val_num = len(validate_dataset)

    # load data to validate dataset
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=num_workers)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # %% LOAD & MODIFY MODEL

    # create model
    net = MobileNetV2(num_classes=7)

    # load pretrain weights
    # download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    """
    model_weight_path = "./mobilenet_v2_pretrain.pth"
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location=device)

    # delete classifier weights
    pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

    # freeze features weights
    for param in net.features.parameters():
        param.requires_grad = False
    """

    # copy the data to GPU
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # optimizer: a module determine the update direction, speed & amount of parameters
    # help the model converge
    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    # %% TRAIN

    #
    best_acc = 0.0
    save_path = './CMN_b16e24_pytorch_v2.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval().cuda()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
            x = torch.randn(1, 3, 224, 224, requires_grad=True,device='cuda')
            torch.onnx.export(net,
                              x,
                              "CNM_b16e24_pytorch_v2.onnx",
                              export_params=True,
                              opset_version=11,
                              do_constant_folding=True,
                              input_names=['input'],
                              output_names=['softmax'])

    print('Finished Training')


if __name__ == '__main__':
    main()
