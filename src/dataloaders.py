from glob import glob
import os
from typing import List


# COCO8の画像ディレクトリパス
COCO_IMAGE_TRAIN_DIR = "datasets/coco8/images/train"
COCO_IMAGE_VAL_DIR   = "datasets/coco8/images/val"

# HomeObjects-3Kの画像ディレクトリ
COCO_IMAGE_VAL_DIR = "datasets/homeobjects-3K/images/val"


def load_coco8_image_path_list() -> List[str]:
    """ COCO8データセットの画像パスを取得するメソッド """
    image_path_list = []
    
    # train
    train_images = glob(os.path.join(COCO_IMAGE_TRAIN_DIR, "*.jpg"))
    image_path_list.extend(train_images)
    
    # val
    val_images = glob(os.path.join(COCO_IMAGE_VAL_DIR, "*.jpg"))
    image_path_list.extend(val_images)

    return image_path_list


def load_homeobjects_val_image_path_list(num_data: int) -> List[str]:
    """ HomeObjects-3K データセットの画像パスを取得するメソッド """
    image_path_list = []
    
    val_images = glob(os.path.join(COCO_IMAGE_VAL_DIR, "*.jpg"))
    image_path_list.extend(val_images)

    return image_path_list[:num_data]
