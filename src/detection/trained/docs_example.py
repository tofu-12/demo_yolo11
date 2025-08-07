from ultralytics import YOLO


# 実行デバイス
DEVICE = "mps"


def docs_train_demo():
    """ ドキュメントの訓練コード """
    # モデルの読み込み
    model = YOLO("yolo11n.pt")

    # COCO8での訓練
    results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=DEVICE)


def docs_pred_demo():
    """ ドキュメントの予測のコード """
    # モデルの読み込み
    model = YOLO("yolo11n.pt")

    # 物体検出
    results = model("https://ultralytics.com/images/bus.jpg")
    results[0].show()

    # 結果の確認
    for result in results:
        xywh  = result.boxes.xywh   # center-x, center-y, width, height
        xywhn = result.boxes.xywhn  # normalized
        xyxy  = result.boxes.xyxy   # top-left-x, top-left-y, bottom-right-x, bottom-right-y
        xyxyn = result.boxes.xyxyn  # normalized

        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        confs = result.boxes.conf  # confidence score of each box

if __name__ == "__main__":
    docs_train_demo()
    docs_pred_demo()
