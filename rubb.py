def rubb_tackle_x_ray_object_dataset():
    # old path
    #old_path = Path(r"D:\download\OPIXray")
    old_path = Path(r"D:\document\awcode\yolov5-3.1\OPIXray")
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