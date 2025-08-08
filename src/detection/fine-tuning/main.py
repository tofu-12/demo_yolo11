import os

from ultralytics import YOLO


# 保存先設定
RESULT_DIR_PATH = "results/detection/fine-tuning"

# 実行デバイス
DEVICE = "mps"


def fine_tuning() -> YOLO:
    """ 
    モデルをHomeObjects-3Kでファインチューニングするメソッド
      
    Returns:
        YOLO: ファインチューニング済みモデル
    """
    # モデルの読み込み
    model = YOLO("yolo11n.pt")

    # データオーグメンテーションの設定
    data_augmentation_config = {
        # カラー変換
        "hsv_h": 0.01,  # 色相調整
        "hsv_s": 0.0,   # 彩度調整
        "hsv_v": 0.5,   # 明るさ調整

        # 幾何変換
        "degrees"    : 0.0,   # 回転 (0.0 ~ 180)
        "translate"  : 0.25,  # 画像をシフト
        "scale"      : 0.5,   # 拡大縮小
        "shear"      : 5,     # x軸とy軸の両方に沿って歪ませる
        "perspective": 0.0,   # 遠近法変換 

        # 反転
        "flipud": 0.0,  # 上下反転
        "fliplr": 0.5,  # 左右反転

        # その他
        "bgr"   : 0.0,  # カラーチャネルをRGBからBGRにスワップ
        "mosaic": 1.0,  # 4つのトレーニング画像を1つに結合
        "mixup" : 0.0,  # 与えられた確率で2つの画像とそのラベルをブレンド
        "cutmix": 0.0,  # ある画像から長方形の領域を切り取り、指定された確率で別の画像に貼り付け
    }

    # ファインチューニング
    model.train(
        # 画像データ
        data="HomeObjects-3K.yaml", 

        # 訓練設定
        epochs       = 10,       # エポック数
        close_mosaic = 3,        # モザイク処理を行うエポック数
        batch        = 16,       # バッチサイズ
        optimizer    = "AdamW",  # 最適化手法
        dropout      = 0.3,      # dropout rate

        imgsz  = 640,     # 画像サイズ　tupleでも指定可
        device = DEVICE,  # 使用デバイス

        # 保存設定
        save     = True,             # 保存するかのフラグ
        project  = RESULT_DIR_PATH,  # 結果の保存先のディレクトリ
        name     = "train",          # 結果の保存先のサブディレクトリ名
        exist_ok = True,             # 結果の保存先が存在する場合は上書きする

        # データオーグメンテーション
        **data_augmentation_config
    )

    return model


if __name__ == "__main__":
    model = fine_tuning()
    model.save(os.path.join(RESULT_DIR_PATH, "fine-tuned_model.pt"))
