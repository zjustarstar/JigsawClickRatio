import pandas
import requests
import os
import time
import multiprocessing


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


def download_img(args):
    i, pic_id, pic_url, savepath = args
    id = pic_id[i]
    image_name = id + '.jpg'
    image_name = os.path.join(savepath, image_name)
    sketch_url = pic_url[i]

    if str(sketch_url) != 'nan':
        # 已经有的不再下载
        if os.path.exists(image_name):
            print(f'{i + 1}/{len(pic_id)}, {image_name} already exists...')
            return

        print(f'downloading {i + 1}/{len(pic_id)}')
        img = crawl_data(sketch_url)
        if img != -1:
            with open(image_name, 'wb') as fp:
                fp.write(img.content)

        time.sleep(10)


if __name__ == '__main__':
    book = pandas.read_csv('./file/ios_click.csv', encoding="utf-8")
    savepath = "F://data//乐信//Jigsaw//JigsawClickIOS"

    pic_id = book['pic_id']
    pic_url = book['picture_url']

    # 创建一个进程池
    pool = multiprocessing.Pool(2)
    # 要处理的数据
    data1 = list(range(0, len(pic_id)))

    # 使用map进行并行处理
    pool.map(download_img,
             [(i, pic_id, pic_url, savepath) for i in data1])

    # 关闭进程池
    pool.close()
    pool.join()
