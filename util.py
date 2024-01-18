import csv
import matplotlib.pyplot as plt


def draw_lines(pred, label):
    # 创建x轴坐标
    x = range(len(pred))
    # 绘制y1的曲线
    plt.plot(x, pred, label='pred')
    # 绘制y2的曲线
    plt.plot(x, label, label='label')

    # 添加图例
    plt.legend()

    # 添加x轴和y轴标签
    plt.xlabel('x')
    plt.ylabel('y')

    # 显示图形
    plt.show()


def pos_cls(pos):
    for i in range(len(pos)):
        if 1 <= pos[i] <= 4:
            pos[i] = 0
        elif 5 <= pos[i] <= 8:
            pos[i] = 1
        else:
            pos[i] = 2
    return pos


def save_to_cvs(img_name, pred, label, pos, savefile, pred0, pred1, pred2):
    pos = pos_cls(pos)
    # 将数据写入CSV文件
    with open(savefile, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['name', 'pos', 'pred', 'true', 'pred0', 'pred1', 'pred2'])  # 写入表头
        for i in range(len(img_name)):
            writer.writerow([img_name[i], pos[i], pred[i], label[i], pred0[i], pred1[i], pred2[i]])  # 逐行写入数据

    print("save done")





