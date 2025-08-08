import os
import sys
from typing import List

from ultralytics import YOLO

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from dataloaders import load_homeobjects_val_image_path_list
from utils import save_result_to_json


# 保存先設定
RESULT_DIR_PATH       = "results/detection/fine-tuning"
RESULT_BOXES_DIR_PATH = "results/detection/fine-tuning/boxes"

# 実行デバイス
DEVICE = "mps"


def pred_homeobjects():
    """ HomeObjects-3Kデータセットに対する予測を行うメソッド """
    # モデルの読み込み
    model = YOLO(os.path.join(RESULT_DIR_PATH, "fine-tuned_model.pt"))

    # HomeObjects-3Kの画像パス
    image_path_list: List = load_homeobjects_val_image_path_list(num_data=10)

    # 物体検出
    results = model.predict(
        # 画像パス
        source = image_path_list,

        # 予測設定
        imgsz  = 640,     # 画像サイズ　tupleでも指定可
        conf   = 0.25,    # 検出の閾値
        device = DEVICE,  # 使用デバイス

        # 保存設定
        save     = True,             # 保存するかのフラグ
        project  = RESULT_DIR_PATH,  # 結果の保存先のディレクトリ
        name     = "images",         # 結果の保存先のサブディレクトリ名
        exist_ok = True              # 結果の保存先が存在する場合は上書きする
    )
    
    # 結果の保存
    for result in results:
        save_result_to_json(result, RESULT_BOXES_DIR_PATH)


if __name__ == "__main__":
    pred_homeobjects()
