import torch.nn
import torch.nn as nn
import numpy as np
import PIL.Image as Image
from torchvision import transforms
import torchvision.models as models
import timm
import clip


class OutputBlock18(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(OutputBlock18, self).__init__()
        self.out = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(mid_ch, mid_ch//2, 3, 1, 1),
            nn.BatchNorm2d(mid_ch//2),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(mid_ch//2, mid_ch//4, 3, 1, 1),
            nn.BatchNorm2d(mid_ch//4),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(8*7*7, mid_ch//2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(mid_ch//2, out_ch),
            nn.Dropout(0.1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.out(x)
        # batch
        b = x.size(0)
        x = x.view(b, -1)
        x = self.fc(x)
        return x


class OutputBlock50(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(OutputBlock50, self).__init__()
        self.out = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(mid_ch, mid_ch//2, 3, 1, 1),
            nn.BatchNorm2d(mid_ch//2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(mid_ch//2, mid_ch//2, 3, 1, 1),
            nn.BatchNorm2d(mid_ch//2),
            nn.ReLU(inplace=True)
            # nn.Sigmoid() 如果使用sigmoid，值被控制在0-1区间，最后计算损失的结果需要乘5，以变为0-5范围以内的数值
        )
        self.fc = nn.Sequential(
            nn.Linear(4096, mid_ch),
            nn.ReLU(inplace=True),
            nn.Linear(mid_ch, out_ch)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.out(x)
        b = x.size(0)
        x = x.view(b, -1)
        x = self.fc(x)
        return x


class TimeModel(nn.Module):
    def __init__(self, model_type="resnet18"):
        super(TimeModel, self).__init__()

        # if model_type == "resnet18":  # 使用resnet除了倒数两层以外的网络
        #     self.model = models.resnet18(pretrained=False)
        #     self.base_model = nn.Sequential(*list(self.model.children()))[: -2]
        #     self.output_model = OutputBlock(512, 128, 1)
        if model_type == "resnet18":  # 使用resnet前五层网络
            self.model = models.resnet18(pretrained=True)
            self.model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=True)
            self.base_model = nn.Sequential(*list(self.model.children()))[: 5]
            self.output_model = OutputBlock18(64, 32, 1)

        elif model_type == "resnet50":
            self.model = models.resnet101(pretrained=True)
            self.model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.base_model = nn.Sequential(*list(self.model.children()))[: -2]  # 使用resnet除了倒数两层以外的网络
            self.output_model = OutputBlock50(2048, 512, 1)

        elif model_type == "vit384":
            self.model = timm.create_model('vit_base_patch32_384',
                                           checkpoint_path='./checkpoints/vit384/pytorch_model.bin', pretrained=True)
            self.model.head = nn.Linear(self.model.head.in_features, 1)
            self.model.blocks = torch.nn.Sequential(*(list(self.model.children())[0:3]))  # 只取前三个block
            # # 最终输出的特征维度是1
            # self.model.blocks.add_module("final_layer", nn.Linear(768, 1))
            # self.model.blocks[0].mlp.drop = torch.nn.Dropout(p=0.3, inplace=False)
            # 第一层增加dropout，防止过拟合
            self.model.blocks[0] = nn.Sequential(
                            nn.Linear(768, 1024),
                            nn.Dropout(p=0.1),
                            nn.GELU(),
                            nn.Linear(1024, 768),
                            nn.Dropout(p=0.1)
            )
        elif model_type == "vit224":
            self.model = timm.create_model('vit_small_patch16_224',
                                           pretrained=True, in_chans=4)
            self.model.head = nn.Linear(self.model.head.in_features, 1)
            # 冻结前三个block
            # for param in self.model.blocks[:2].parameters():
            #     param.requires_grad = False

            # # 最终输出的特征维度是1
            # self.model.blocks.add_module("final_layer", nn.Linear(768, 1))
            # self.model.blocks[0].mlp.drop = torch.nn.Dropout(p=0.3, inplace=False)
            # 第一层增加dropout，防止过拟合
            # self.model.blocks[0] = nn.Sequential(
            #                 nn.Linear(768, 1024),
            #                 nn.Dropout(p=0.1),
            #                 nn.GELU(),
            #                 nn.Linear(1024, 768),
            #                 nn.Dropout(p=0.1)
            # )
        # vit-b-32 输出维度是512， vit-l-14输出维度是768, RN50是1024
        elif model_type == 'clip':
            self.model, _ = clip.load("./checkpoints/clip/vit-B-32.pt")
            self.fc = nn.Sequential(
                nn.Linear(513, 128),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.Dropout(0.3),
                nn.GELU(),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.Dropout(0.1),
                nn.GELU(),
                nn.Linear(32, 1),
            )
        else:
            raise Exception("Please select a model")

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)


    def pos_cls(self, pos):
        for i in range(len(pos)):
            if 1 <= pos[i] <= 4:
                pos[i] = 0
            elif 5 <= pos[i] <= 8:
                pos[i] = 1
            else:
                pos[i] = 2
        return pos


    def forward(self, x, pos, model_type="resnet18"):
        if model_type == "resnet18":
            pos = self.pos_cls(pos)
            pos_array = pos.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            pos_info = torch.ones_like(x[:,:1,:,:]) * pos_array
            x = torch.cat((x, pos_info), dim=1)
            x = self.base_model(x)
            # x = self.avg_pool(x)
            x = self.output_model(x)
            return x

        elif model_type == "resnet50":
            pos_array = pos.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            pos_info = torch.ones_like(x[:, :1, :, :]) * pos_array
            x = torch.cat((x, pos_info), dim=1)
            x = self.base_model(x)
            # batch, ch = x.size(0), x.size(1)
            # x = self.avg_pool(x).view(batch, ch)
            # x = self.avg_pool(x)
            x = self.output_model(x)
            batch, ch = x.size(0), x.size(1)
            x = x.view(batch, ch)
            return x

        elif model_type == "model":
            x = self.base_model(x)
            x = self.max_pool(x)
            x = self.output_model(x)
            batch, ch = x.size(0), x.size(1)
            x = x.view(batch, ch)
            return x
        elif model_type == 'vit384':
            x = self.model(x)
            return x
        elif model_type == 'vit224':
            pos = self.pos_cls(pos)
            pos_array = pos.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            pos_info = torch.ones_like(x[:, :1, :, :]) * pos_array
            x = torch.cat((x, pos_info), dim=1)
            x = self.model(x)
            return x
        elif model_type == 'clip':
            batch, ch = x.size(0), x.size(1)
            # ori_pos = pos.unsqueeze(1)
            pos = self.pos_cls(pos).unsqueeze(1)
            pos_e = pos.expand(batch, 768)
            with torch.no_grad():
                img_fe = self.model.encode_image(x)
            # img_fe =img_fe + pos_e
            img_fe = torch.cat((img_fe, pos), dim=-1)
            x = self.fc(img_fe.float())
            return x
        else:
            raise Exception("Please select a model")


