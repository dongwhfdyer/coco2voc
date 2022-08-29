import glob
import os
import random
import shutil
import time
from pathlib import Path

from PIL import Image
from cv2 import cv2
import matplotlib.image as mi
import matplotlib.pyplot as plt
import numpy as np
import png  # https://pypng.readthedocs.io/en/latest/


def read_voc_segmentation_label_img():
    label_img_path = r'D:\ANewspace\code\deeplabv3-plus-pytorch\datasets\pighead\SegmentationClass\class_labels\0001.png'
    original_img_path = r"D:\ANewspace\code\deeplabv3-plus-pytorch\datasets\pighead\JPEGImages\0001.jpg"
    img = Image.open(label_img_path).convert("P")
    img.save('rubb/rubb.png')

    # Convert_Palette(original_img_path, label_img_path)


def visual_colormap():
    """
    It's used for visualizing the color map of the segmentation result.
    :return:
    """
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


def extract_pig_face_mask(imgs_folder, segs_folder, new_imgs_folder):
    """
    It will extract the mask of the pig face from the original image, and save it in a new folder.
    :return:
    """
    # images_folder = Path(r"D:\ANewspace\code\deeplabv3-plus-pytorch\datasets\pighead\JPEGImages")
    # seg_folder = Path(r"D:\ANewspace\code\deeplabv3-plus-pytorch\datasets\pighead\SegmentationClass")
    # new_img_folder = Path(r"D:\ANewspace\code\deeplabv3-plus-pytorch\datasets\pighead\JPEGImages_pig_face")

    imgs_folder = Path(imgs_folder)
    segs_folder = Path(segs_folder)
    new_imgs_folder = Path(new_imgs_folder)

    for img_path in imgs_folder.glob("*.jpg"):
        img_name = img_path.name
        seg_path = segs_folder / (img_name.replace(".jpg", ".png"))
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
            new_img_path = new_imgs_folder / img_name
            cv2.imwrite(str(new_img_path), img)


def extractPigFaceFor_new_agg_face_only_folder():
    new_agg_face_only_folder_path = Path(r"d:\ANewspace\code\pig_face_weight_correlation\datasets\new_agg_face_only")
    exact_face_folder = Path(r"d:\ANewspace\code\pig_face_weight_correlation\datasets\exact_face_only")
    delete_folders(exact_face_folder)
    create_folders(exact_face_folder)
    for id_folder in new_agg_face_only_folder_path.iterdir():
        exactFace_id_folder = exact_face_folder / id_folder.name
        create_folders(exactFace_id_folder)
        for img_path in id_folder.glob("*"):
            img_name = img_path.name
            img = cv2.imread(str(img_path))
            img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            x, y, w, h = cv2.boundingRect(img_gray)
            img = img[y:y + h, x:x + w]
            new_img_path = exactFace_id_folder / img_name
            cv2.imwrite(str(new_img_path), img)


def delete_folders(*folder_path):
    for folder in folder_path:
        if os.path.exists(folder):
            shutil.rmtree(folder)


def create_folders(*folders):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


def selected_600_pig_face():
    """
    It is used for pig face landmark detection. I prepared 600 pig face images for the detection.
    It will select 600 pig face images from the original image folder.
    :return:
    """
    old_path = Path(r"D:\ANewspace\code\deeplabv3-plus-pytorch\datasets\pighead\JPEGImages_pig_face")
    new_path = Path(r"D:\ANewspace\code\deeplabv3-plus-pytorch\datasets\pighead\pig_face_600")
    delete_folders(new_path)
    create_folders(new_path)

    paths = os.listdir(old_path)
    random.shuffle(paths)
    selected_paths = paths[:600]
    for path in selected_paths:
        shutil.copy(str(old_path / path), str(new_path / path))


def rubb_tackle_x_ray_object_dataset():
    # old path
    old_path = Path(r"D:\download\OPIXray")
    train_path = old_path / "train"
    train_imgs_path = train_path / "train_image"
    train_annos_path = train_path / "train_annotation"
    test_path = old_path / "test"
    test_imgs_path = test_path / "test_image"
    test_annos_path = test_path / "test_annotation"

    # new path
    new_path = Path(r"rubb/opixray_yolo")
    new_train_path = new_path / "train"
    new_train_imgs_path = new_train_path / "images"
    new_train_labels_path = new_train_path / "labels"
    new_test_path = new_path / "test"
    new_test_imgs_path = new_test_path / "images"
    new_test_labels_path = new_test_path / "labels"

    delete_folders(new_train_path, new_test_path)
    create_folders(new_train_imgs_path, new_train_labels_path, new_test_imgs_path, new_test_labels_path)
    obj_id_class = {'Straight_Knife': 0, 'Folding_Knife': 1, 'Scissor': 2, 'Utility_Knife': 3, 'Multi-tool_Knife': 4}

    def tackle_one_dataset(img_folder, anno_folder, new_img_folder, new_anno_folder):
        for img_path in img_folder.glob("*.jpg"):
            img_name = img_path.name
            anno_path = anno_folder / (img_name.replace(".jpg", ".txt"))
            anno_file_handle = open(str(anno_path), 'r')
            anno_info = anno_file_handle.readlines()
            new_anno_file_handle = open(str(new_anno_folder / (img_name.replace(".jpg", ".txt"))), 'w')
            for line in anno_info:
                img_name, obj, x1, y1, x2, y2 = line.split()

                img = cv2.imread(str(img_path))
                # get img_width and img_height
                img_width = img.shape[1]
                img_height = img.shape[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # bounding box width and height
                w = x2 - x1
                h = y2 - y1
                # center point of the bounding box
                cx = x1 + w / 2
                cy = y1 + h / 2
                # normalized center point
                nx = cx / img_width
                ny = cy / img_height
                # normalized width and height
                nw = w / img_width
                nh = h / img_height
                # saving two decimal places
                nx = round(nx, 4)
                ny = round(ny, 4)
                nw = round(nw, 4)
                nh = round(nh, 4)

                new_anno_file_handle.write(str(obj_id_class[obj]) + " " + str(nx) + " " + str(ny) + " " + str(nw) + " " + str(nh) + "\n")

                # x1, y1, x2, y2 = x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height

                # # ---------kkuhn-block------------------------------ # draw
                # img = cv2.imread(str(img_path))
                # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.imshow("img", img)
                # cv2.waitKey(0)
                # # ---------kkuhn-block------------------------------
            anno_file_handle.close()
            new_anno_file_handle.close()
            shutil.copyfile(str(img_path), str(new_img_folder / img_name))
            pass

    tackle_one_dataset(train_imgs_path, train_annos_path, new_train_imgs_path, new_train_labels_path)
    tackle_one_dataset(test_imgs_path, test_annos_path, new_test_imgs_path, new_test_labels_path)

    pass
    # test


if __name__ == '__main__':
    # read_voc_segmentation_label_img()
    # rubb_colormap()
    # train_val_split()
    # extract_pig_face_mask()
    # selected_600_pig_face()
    # extractPigFaceFor_new_agg_face_only_folder()
    rubb_tackle_x_ray_object_dataset()
    pass
