import os
import sys
from typing import List

from ultralytics import YOLO

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from dataloaders import load_coco8_image_path_list
from utils import save_result_to_json


# 保存先設定
RESULT_DIR_PATH       = "results/detection/trained"
RESULT_BOXES_DIR_PATH = "results/detection/trained/boxes"

# 実行デバイス
DEVICE = "mps"


def pred_coco():
    """ COCO8データセットに対する予測を行うメソッド """
    # モデルの読み込み
    model = YOLO("yolo11n.pt")

    # COCO8の画像パス
    image_path_list: List = load_coco8_image_path_list()

    # 物体検出
    results = model(
        # 画像パス
        source = image_path_list,

        # 予測設定
        imgsz  = 640,     # 画像サイズ　tupleでも指定可
        conf   = 0.25,    # 検出の閾値
        device = DEVICE,  # 使用デバイス

        # 保存設定
        save    = True,             # 保存するかのフラグ
        project = RESULT_DIR_PATH,  # 結果の保存先のディレクトリ
        name    = "images"          # 結果の保存先のサブディレクトリ名
    )
    
    # 結果の保存
    for result in results:
        save_result_to_json(result, RESULT_BOXES_DIR_PATH)


if __name__ == "__main__":
    pred_coco()
