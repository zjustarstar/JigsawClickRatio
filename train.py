import torch
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
from dataset import JigsawImgDataSet
from torch.utils.data import DataLoader
import networks
import argparse
import logging
import os
from pytorchtools import EarlyStopping

MODEL_TYPE = "clip"


# parameters setting
parser = argparse.ArgumentParser(description='TimePrediction')
# Datasets
parser.add_argument('--dataset_train',
                    default='./file/train.json', type=str)
parser.add_argument('--dataset_test',
                    default='./file/test.json', type=str)
# Optimization options
parser.add_argument('--epochs', default=1000000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--train_batch', default=256, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test_batch', default=800, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--lr', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
# Checkpoints
parser.add_argument('--checkpoints_dir', default='./checkpoints/', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--test_interval', default=1, type=int, metavar='N',
                    help='test every X interval')
parser.add_argument('--save_interval', default=30, type=int, metavar='N',
                    help='save every X interval ')

# Architecture
parser.add_argument('--model_name', default=MODEL_TYPE, type=str)

# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()


trainset = JigsawImgDataSet(dataset_path=args.dataset_train, model=MODEL_TYPE)
trainloader = DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=0)
print(f"train samples size:{len(trainloader.dataset)}")
testset = JigsawImgDataSet(dataset_path=args.dataset_test, model=MODEL_TYPE)
testloader = DataLoader(testset, batch_size=args.test_batch, shuffle=True, num_workers=0)
print(f"test samples size:{len(testloader.dataset)}")
dataloaders = {'train': trainloader, 'val': testloader}


def init_logger(checkpoints_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    handler = logging.FileHandler(f"./{checkpoints_dir}/log.txt")
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def write_log_train_test(logger, loss, epoch, test_accuracy_60=0, test_accuracy_120=0, test_accuracy_180=0, mode='TRAIN'):
    if mode == "TRAIN":
        log = f'[{mode}]  epoch: %3d  MAE loss: %.3f  accuracy (mae<=0.01:%.3f  mae<=0.02:%.3f  mae<=0.03:%.3f)' % \
              (epoch, loss, test_accuracy_60, test_accuracy_120, test_accuracy_180)
        logger.info(log)
    else:
        log = f'[{mode}]  epoch: %3d  MAE loss: %.3f  accuracy (mae<=0.01:%.3f  mae<=0.02:%.3f  mae<=0.03:%.3f)' % \
              (epoch, loss, test_accuracy_60, test_accuracy_120, test_accuracy_180)
        logger.info(log)


def write_log(logger, loss, epoch, test_accuracy_5=0, test_accuracy_10=0, test_accuracy_20=0, mode='TRAIN'):
    if mode == "TRAIN":
        log = f'[{mode}] epoch: %3d  loss: %.3f' % (epoch, loss)
        logger.info(log)
    else:
        log = f'[{mode}]  epoch: %3d  MAE loss: %.3f  accuracy (mae<=50:%.3f  mae<=100:%.3f  mae<=200:%.3f)' % \
              (epoch, loss, test_accuracy_5, test_accuracy_10, test_accuracy_20)
        logger.info(log)


def train_model(model, model_type, criterion, optimizer, exp_lr_scheduler, early_stopping, save_interval, test_interval, checkpoints_dir, num_epochs):

    logger = init_logger(checkpoints_dir)
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_img_num = 0

        true_num_60 = 0
        true_num_120 = 0
        true_num_180 = 0

        for img, pos, labels_class, labels in dataloaders['train']:
            if torch.cuda.is_available():
                labels = labels.cuda()
                pos = pos.cuda()
                img = img.cuda()

            # loss
            results = model(img, pos, model_type=model_type)
            results = torch.squeeze(results, 1)
            results = results.float()
            labels = labels.float()
            loss = criterion(results, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            exp_lr_scheduler.step()

            train_loss += loss.data * labels.size(0)
            train_img_num += labels.size(0)

            results = results.squeeze(-1).detach().cpu().numpy()
            labels = labels.detach().squeeze().cpu().numpy()

            temp = np.abs(results - labels)
            num_60 = np.sum(temp <= 1)
            true_num_60 += num_60
            num_120 = np.sum(temp <= 2)
            true_num_120 += num_120
            num_180 = np.sum(temp <= 3)
            true_num_180 += num_180


        # 训练准确率
        average_train_loss = train_loss / train_img_num
        test_accuracy_60 = true_num_60 / train_img_num
        test_accuracy_120 = true_num_120 / train_img_num
        test_accuracy_180 = true_num_180 / train_img_num

        write_log_train_test(logger, average_train_loss, epoch+1, test_accuracy_60, test_accuracy_120, test_accuracy_180, mode='TRAIN')

        if (epoch+1) % test_interval == 0:
            test_loss = 0
            test_img_num = 0
            true_num_60 = 0
            true_num_120 = 0
            true_num_180 = 0

            for img, pos, labels_cls, labels in dataloaders['val']:
                if torch.cuda.is_available():
                    labels = labels.cuda()
                    pos = pos.cuda()
                    img = img.cuda()

                with torch.no_grad():
                    results = model(img, pos, model_type=model_type)
                    results = torch.squeeze(results, 1)
                    results = results.float()
                    # print(results)
                    # results = results.squeeze(-1)
                    # print(results.size(), labels.size())

                    test_loss += nn.L1Loss()(results, labels).data * labels.size(0)
                    test_img_num += labels.size(0)
                    results = results.squeeze(-1).detach().cpu().numpy()
                    labels = labels.detach().squeeze().cpu().numpy()
                    temp = np.abs(results - labels)

                    num_60 = np.sum(temp <= 1)
                    true_num_60 += num_60
                    num_120 = np.sum(temp <= 2)
                    true_num_120 += num_120
                    num_180 = np.sum(temp <= 3)
                    true_num_180 += num_180

            # 测试准确率
            average_test_loss = test_loss / test_img_num
            test_accuracy_60 = true_num_60 / test_img_num
            test_accuracy_120 = true_num_120 / test_img_num
            test_accuracy_180 = true_num_180 / test_img_num
            write_log_train_test(logger, average_test_loss, epoch+1, test_accuracy_60, test_accuracy_120, test_accuracy_180, mode='TEST')

            early_stopping(average_test_loss, model)
            # 若满足 early stopping 要求
            if early_stopping.early_stop:
                print(f"MAE = {early_stopping.val_loss_min}")
                print("Early stopping")
                # 结束模型训练
                break

        # print("-" * 150)


if __name__ == '__main__':
    print("Model :", args.model_name)
    args.checkpoints_dir = os.path.join(args.checkpoints_dir, args.model_name)
    if not os.path.exists(args.checkpoints_dir):
        os.mkdir(args.checkpoints_dir)
    model = networks.TimeModel(model_type=args.model_name)
    if torch.cuda.is_available():
        model.cuda()
    model.train()

    # p = os.path.join(args.checkpoints_dir, param_name)
    # if os.path.exists(p):
    #     print(f'load existing model:{p}')
    #     model.load_state_dict(torch.load(p))

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.95)
    early_stopping = EarlyStopping(patience=12, verbose=True, path=os.path.join(args.checkpoints_dir, "best_checkpoint.pth"))
    train_model(model, args.model_name, criterion, optimizer, exp_lr_scheduler, early_stopping, args.save_interval, args.test_interval, args.checkpoints_dir, args.epochs)

