import pandas
import requests
import os
import time
from multiprocessing import Pool


def crawl_data(url):
    try:
        #response = requests.get(url)
        img = requests.get(url, timeout=(5, 7))
        # 在这里处理网页数据的逻辑
        return img
    except requests.exceptions.Timeout:
        print("连接超时，等待一段时间后重新连接...")
        time.sleep(10)  # 等待5秒后重新连接
        return crawl_data(url)
    except requests.exceptions.RequestException as e:
        print("连接失败:", str(e))
        #return None
        return -1

book = pandas.read_csv('./file/ios_click.csv', encoding="utf-8")
savepath = "D://LexinData//jigsawClickIOS"

pic_id = book['pic_id']
pic_url = book['picture_url']
for i in range(2000, len(pic_id)):
    id = pic_id[i]
    image_name = id + '.jpg'
    image_name = os.path.join(savepath, image_name)
    sketch_url = pic_url[i]

    if str(sketch_url) != 'nan':
        # 已经有的不再下载
        if os.path.exists(image_name):
            print(f'{i+1}/{len(pic_id)}, {image_name} already exists...')
            continue

        img = crawl_data(sketch_url)
        if img != -1:
            with open(image_name, 'wb') as fp:
                fp.write(img.content)

        time.sleep(10)

    print(f'downloaded {i+1}/{len(pic_id)}')
