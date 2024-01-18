from torch.utils.data import Dataset
import cv2
import json
import torch
from PIL import Image
import numpy as np
import os
import clip


from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
_, clip_preprocess = clip.load("./checkpoints/clip/ViT-B-32.pt", device=device)


transform_resnet_image = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

transform_vit384 = transforms.Compose([
            transforms.Resize([384,384]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

transform_vit224 = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

# clip 模型的resize到224或者384都可以
transforms_clip = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
        ])


class JigsawImgDataSet(Dataset):
    """
    data_path:包含所有的图片的路径和标签
    """
    # 初始化
    def __init__(self, dataset_path="./file/train.json",
                 img_root_path="D://myproject//JigsawClickAndroid",
                 model='resnet'):
        self.img_name = []
        self.click_ratio = []
        self.img_pos = []
        self.model = model
        self.img_rootpath = img_root_path

        with open(dataset_path, "r") as r:
            data = json.load(r)

        for key, value in data.items():
            fullname = os.path.join(self.img_rootpath, key)
            self.img_name.append(fullname)
            self.img_pos.append(value[0])
            self.click_ratio.append(value[1])

        self.transforms = 0
        if model == 'vit384':
            self.transforms = transform_vit384
        elif model == 'vit224':
            self.transforms = transform_vit224
        elif model == 'clip':
            self.transforms = clip_preprocess
        else:
            self.transforms = transform_resnet_image

    def __len__(self):
        return len(self.img_name)

    def get_label_class(self, label):
        # label分类
        label_class = 0
        if label > 13:
            label_class = 2
        elif label>9:
            label_class = 1

        return label_class

    def __getitem__(self, index):
        img_name = self.img_name[index]
        pos = self.img_pos[index]
        label = self.click_ratio[index]
        img_path = os.path.join(self.img_rootpath, img_name)

        lable_class = self.get_label_class(label=label)

        default_size = 224
        if self.model == 'vit384':
            default_size = 384
        elif self.model == 'vit224':
            default_size = 224
        try:
            im = Image.open(img_path)
            if im.mode == 'RGBA':
                im = np.array(im)
                rgb_info = im[:, :, :3]
                a_info = im[:, :, 3]
                rgb_info[a_info == 0] = [255, 255, 255]
                im = Image.fromarray(rgb_info)

            img = im.convert('RGB')
            if self.transforms:
                try:
                    img = self.transforms(img)
                except Exception as e:
                    print("Cannot transform image: {}".format(img_name))
        except IOError as e:
            print(str(e))
            print(f"fail to open image {img_name}")
            return torch.zeros((3,default_size,default_size)), 0, 0

        return img, pos, lable_class, label
