
import json
import pandas
import random
import os
import tqdm
import numpy as np

# jpg_file = "D://LexinData//JigsawClickAndroid"
jpg_file = "F://data//乐信//Jigsaw//JigsawClickAndroid"
cvs_file_path = ".//file//android_click.csv"


def get_stat_info(data):
    ave_val = np.mean(np.array(data)) # 79.09247191011237
    max_val = np.max(np.array(data))
    min_val = np.min(np.array(data))
    std_val = np.std(np.array(data))  # 24.377676506738464
    return ave_val, max_val, min_val, std_val


def get_all_data_info(cvs_file_path, whole_json_path):
    book = pandas.read_csv(cvs_file_path, encoding="gb2312")
    date = book['release_date'].values.tolist()
    pos = book['rank'].values.tolist()
    pic_id = book['pic_id'].values.tolist()
    show_num = book['pic_unique_show'].values.tolist()
    click_ratio = book['perc'].values.tolist()

    # temp = click_ratio
    # ratio = [i.replace("%", "") for i in temp]
    # ratio = [round(float(i), 2) for i in ratio]
    # # ave=12.7, min=3.9, max=31.22, stdv=3.24
    # avev, maxv, minv, stdv = get_stat_info(ratio)

    #保存信息
    data = {}
    for i in tqdm.tqdm(range(len(pic_id))):
        filename = os.path.join(jpg_file, pic_id[i] + ".jpg")
        if not os.path.exists(filename):
            continue

        ratio = click_ratio[i].replace("%", "")
        ratio = round(float(ratio), 2)
        data[pic_id[i] + ".jpg"] = [pos[i], ratio]

    with open(whole_json_path, "w") as f:
        json.dump(data, f)

    return data


# 将数据分别写入train.json和test.json文件
def create_train_test_json(data, ratio=0.8):
    train_json_path = "./file/train.json"
    test_json_path = "./file/test.json"
    train_data = {}
    test_data = {}

    with open(train_json_path, "w") as f_train:
        with open(test_json_path, "w") as f_test:
            for key, value in data.items():
                random_num = random.random()
                if random_num < ratio:
                    train_data[key] = value
                else:
                    test_data[key] = value
                    # train_data[data[i]]
            json.dump(train_data, f_train)
            json.dump(test_data, f_test)

    print("训练集大小:", len(train_data))
    print("测试集大小:", len(test_data))


if __name__ == '__main__':
    save_whole_json_path = "./file/whole_original.json"
    data = get_all_data_info(cvs_file_path, save_whole_json_path)
    # 90%的训练数据
    create_train_test_json(data, 0.9)
