import glob
import os
import random
import shutil
from pathlib import Path

from PIL import Image
from cv2 import cv2
import matplotlib.image as mi
import matplotlib.pyplot as plt
import numpy as np
import png  # https://pypng.readthedocs.io/en/latest/


# 使用PIL库生成调色板格式的图
def Convert_Palette(img_png, label_img):
    png_data = png.Reader(label_img)
    # print(png_data.read()) 这里可以获取该png图片的信息(tuple格式储存的)，如果是调色板图片，还可以获取调色板
    voc_palette = png_data.read()[3]['palette']  # 得到voc的label的调色板(list形式储存的)
    palette = np.array(voc_palette)  # 256*3
    palette = palette.reshape(256, 1, 3).astype(np.uint8)  # 256*1*3 把int32改成uint8(opencv中储存图像的数据格式 0 - 256)

    out_img = Image.open(img_png)
    out_img.putpalette(palette)  # 就是以刚刚得到的调色板将图片转换为调色板模式的伪彩图
    out_img.save('rubb/rubb.png')  # 保存为png格式的伪彩图


def read_voc_segmentation_label_img():
    label_img_path = r'D:\ANewspace\code\deeplabv3-plus-pytorch\datasets\pighead\SegmentationClass\class_labels\0001.png'
    original_img_path = r"D:\ANewspace\code\deeplabv3-plus-pytorch\datasets\pighead\JPEGImages\0001.jpg"
    img = Image.open(label_img_path).convert("P")
    img.save('rubb/rubb.png')

    # Convert_Palette(original_img_path, label_img_path)


def rubb_colormap():
    # "
    img_path = r"D:\ANewspace\code\deeplabv3-plus-pytorch\datasets\pighead\SegmentationClass\class_labels\0001.png"
    ss = Image.open(img_path)
    # ss = Image.open('rubb/rubb.png')
    ss_numpy = np.array(ss)

    bin_colormap = np.random.randint(0, 255, (256, 3))
    bin_colormap = bin_colormap.astype(np.uint8)
    visualimg = Image.fromarray(ss_numpy, "P")
    palette = bin_colormap  # long palette of 768 items
    visualimg.putpalette(palette)
    vis_name = "rubb/rubb_1.png"
    visualimg.save(vis_name, format='PNG')

    # files = [
    #         'SegmentationObject/2007_000129.png',
    #         'SegmentationClass/2007_000129.png',
    #         'SegmentationClassRaw/2007_000129.png', # processed by _remove_colormap()
    #                                                 # in captainst's answer...
    #         ]

    print('\nfile: {}\nanno: {}\nimg info: {}'.format(
        img_path, set(ss_numpy.flatten()), ss))


def train_val_split():
    img_list_txt_path = r"D:\ANewspace\code\deeplabv3-plus-pytorch\datasets\pighead\SegmentationClass\images_ids.txt"
    train_val_ratio = 0.8
    with open(img_list_txt_path, 'r') as f:
        img_list = f.readlines()

    random.shuffle(img_list)
    train_list = img_list[:int(len(img_list) * train_val_ratio)]
    val_list = img_list[int(len(img_list) * train_val_ratio):]
    with open(r"D:\ANewspace\code\deeplabv3-plus-pytorch\datasets\pighead\train.txt", 'w') as f:
        for line in train_list:
            f.write(line)
    with open(r"D:\ANewspace\code\deeplabv3-plus-pytorch\datasets\pighead\val.txt", 'w') as f:
        for line in val_list:
            f.write(line)


def extract_pig_face_mask():
    images_folder = Path(r"D:\ANewspace\code\deeplabv3-plus-pytorch\datasets\pighead\JPEGImages")
    seg_folder = Path(r"D:\ANewspace\code\deeplabv3-plus-pytorch\datasets\pighead\SegmentationClass")

    new_img_folder = Path(r"D:\ANewspace\code\deeplabv3-plus-pytorch\datasets\pighead\JPEGImages_pig_face")

    for img_path in images_folder.glob("*.jpg"):
        img_name = img_path.name
        seg_path = seg_folder / (img_name.replace(".jpg", ".png"))
        if seg_path.exists():
            img = cv2.imread(str(img_path))
            seg = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
            seg_unique = np.unique(seg)  # 0,1
            mask = seg == 0
            img[mask] = 0
            # get the bounding box of the pig face
            x, y, w, h = cv2.boundingRect(seg)
            # crop the pig face
            img = img[y:y + h, x:x + w]
            # resize the pig face
            # img = cv2.resize(img, (256, 256))
            # save the pig face
            new_img_path = new_img_folder / img_name
            cv2.imwrite(str(new_img_path), img)

            # cv2.imwrite(str(new_img_folder / img_name), img)


def delete_folders(*folder_path):
    for folder in folder_path:
        if os.path.exists(folder):
            shutil.rmtree(folder)


def create_folders(*folders):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)



def selected_600_pig_face():
    old_path = Path(r"D:\ANewspace\code\deeplabv3-plus-pytorch\datasets\pighead\JPEGImages_pig_face")
    new_path = Path(r"D:\ANewspace\code\deeplabv3-plus-pytorch\datasets\pighead\pig_face_600")
    delete_folders(new_path)
    create_folders(new_path)

    paths = os.listdir(old_path)
    random.shuffle(paths)
    selected_paths = paths[:600]
    for path in selected_paths:
        shutil.copy(str(old_path / path), str(new_path / path))


if __name__ == '__main__':
    # read_voc_segmentation_label_img()
    # rubb_colormap()
    # train_val_split()
    # extract_pig_face_mask()
    selected_600_pig_face()

