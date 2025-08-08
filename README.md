# YOLO11 Detection Project

YOLO11を使用した物体検出プロジェクトです。

## 環境設定

### 依存関係
```bash
uv sync
```

### 実行環境
- Python 3.12.7+
- ultralytics 8.3.175+


## プロジェクト構成

```
yolo11/
├── src/
│   └── detection/
│       ├── trained/          # 事前訓練済みモデルでの推論
│       └── fine-tuning/      # ファインチューニング
├── datasets/                 # ダウンロードされたデータセット
└── results/                  # 出力結果
    └── detection/
        ├── trained/
        │   ├── images/       # 検出結果画像
        │   └── boxes/        # 検出結果JSON
        └── fine-tuning/      # ファインチューニング結果
```

## 使用方法

### 1. 事前訓練済みモデルでの推論
```bash
uv run src/detection/trained/main.py
```
- COCO8データセットの画像に対して物体検出を実行
- 結果は `results/detection/trained/` に保存
- JSON形式で検出結果を出力

### 2. ファインチューニング
```bash
uv run src/detection/fine-tuning/main.py
```
- HomeObjects-3Kデータセットでモデルをファインチューニング
- データオーグメンテーション設定済み
- ファインチューニング済みモデルを保存

## 出力形式

### JSON出力例
```json
[
  {
    "image_name": "sample",
    "detection_id": 0,
    "class_id": 0,
    "class_name": "person",
    "confidence": 0.85,
    "bbox": {
      "xyxy": [100, 50, 200, 150],
      "xywh": [150, 100, 100, 100],
      "xyxyn": [0.2, 0.1, 0.4, 0.3],
      "xywhn": [0.3, 0.2, 0.2, 0.2]
    }
  }
]
```

## 設定

### デバイス設定
- MPS (Apple Silicon): `DEVICE = "mps"`
- CUDA: `DEVICE = "cuda"`
- CPU: `DEVICE = "cpu"`

### データオーグメンテーション
- HSV変換、幾何変換、反転処理など
- `src/detection/fine-tuning/main.py` で設定可能
