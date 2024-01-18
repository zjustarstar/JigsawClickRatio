import os.path

import torch
import numpy as np
import math
import json
from networks import TimeModel
import util
import clip
from torchvision import transforms
from PIL import Image

MODEL_TYPE = 'clip'

img_root = "D://myproject//JigsawClickAndroid"

device = "cuda" if torch.cuda.is_available() else "cpu"
_, clip_preprocess = clip.load("./checkpoints/clip/ViT-B-32.pt", device=device)


transform_resnet_image = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

transform_vit224 = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


def img_process(imgfile, model="resnet"):
    try:
        im = Image.open(imgfile)
        if im.mode == 'RGBA':
            im = np.array(im)
            rgb_info = im[:, :, :3]
            a_info = im[:, :, 3]
            rgb_info[a_info == 0] = [255, 255, 255]
            im = Image.fromarray(rgb_info)

        img = im.convert('RGB')
        if model == "resnet":
            x = transform_resnet_image(img)
        elif model == "clip":
            x = clip_preprocess(img)
        else:
            x = transform_vit224(img)
    except IOError:
        print(f"fail to open image {imgfile}")
        return torch.zeros((3, 224, 224)), 0

    return x


def get_pred(data, model):
    true_num_60 = 0
    true_num_120 = 0
    true_num_180 = 0

    pred, label, img_name = [], [], []
    total_error = 0
    for key, value in data.items():
        fname = key
        pos = torch.from_numpy((np.array(value[0])))
        true_ratio = value[1]

        # batch value
        input_pos = torch.unsqueeze(pos, 0).cuda()
        x = img_process(os.path.join(img_root, fname),model=MODEL_TYPE).cuda()
        x = torch.unsqueeze(x, 0)

        results = model(x, input_pos, model_type=MODEL_TYPE)
        results = results.squeeze(-1)
        results = results.float().item()

        temp = math.fabs(results - true_ratio)

        pred.append(results)
        label.append(true_ratio)
        img_name.append(key)

        if temp < 1:
            true_num_60 += 1
        if temp < 2:
            true_num_120 += 1
        if temp < 3:
            true_num_180 += 1

        total_error += temp

    avg_error = total_error / len(pred)
    num60 = round(true_num_60 / len(pred),2)
    num120 = round(true_num_120 / len(pred),2)
    num180 = round(true_num_180 / len(pred),2)
    print(f'avg = {avg_error}, error<1={num60}, error<2={num120}, error<3={num180}')

    return img_name, pred, label



if __name__ == '__main__':
    # Model & Optimizer Definition
    model_path = './checkpoints/' + MODEL_TYPE + '/best_checkpoint.pth'
    model = TimeModel(model_type=MODEL_TYPE)
    model.cuda()

    ckp = torch.load(model_path)
    model.load_state_dict(ckp)
    model = model.eval()
    print(f"load model {model_path}...")

    test_json_path = "./file/test.json"
    with open(test_json_path, "r") as r:
        data = json.load(r)

    print(f'length of samples = {len(data.items())}')
    img_name, pred, label = get_pred(data, model)
    util.draw_lines(pred, label)
    # 写入文件
    save_file = "pred.csv"
    util.save_to_cvs(img_name, pred, label, save_file)

#
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# ])
#
# max_value = -1
# min_value = 100
#
# count = 0
# for file in os.listdir(test_img_path):
#     img_name = os.path.join(test_img_path, file)
#     img = cv2.imread(img_name)
#
#     img = transform_test(img).cuda()
#     img = torch.unsqueeze(img, 0)  # 给最高位添加一个维度，也就是batchsize的大小
#
#     res = model(img)
#     res = res.detach().cpu().numpy()[0][0]
#     print(img_name, res)
#
#
#     temp = float(img_name.split("_")[4])
#     # print(temp)
#     chazhi_res = np.abs(res-temp)
#     print(chazhi_res)
#     # print(chazhi_res)
#     # raise
#
#     if chazhi_res > max_value:
#         max_value = chazhi_res
#     if chazhi_res < min_value:
#         min_value = chazhi_res
# print(max_value, min_value)




