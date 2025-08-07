import json
import os
from pathlib import Path


def save_result_to_json(result, result_boxes_dir_path: str) -> None:
    """ 1つの結果をJSONファイルに保存するメソッド """
    if result:
        # 画像名の取得
        image_name = Path(result.path).stem

        if result.boxes:
            # boxesの変換
            boxes = result.boxes.cpu().numpy()

            # 結果の作成
            detections = []
            for i in range(len(boxes.cls)):
                detection = {
                    "image_name"  : image_name,
                    "detection_id": i,
                    "class_id"    : int(boxes.cls[i]),
                    "class_name"  : result.names[int(boxes.cls[i])],
                    "confidence"  : float(boxes.conf[i]),

                    "bbox": {
                        "xyxy" : [float(x) for x in boxes.xyxy[i]],   # [x1, y1, x2, y2]
                        "xywh" : [float(x) for x in boxes.xywh[i]],   # [center_x, center_y, width, height]
                        "xyxyn": [float(x) for x in boxes.xyxyn[i]],  # normalized [x1, y1, x2, y2]
                        "xywhn": [float(x) for x in boxes.xywhn[i]]   # normalized [center_x, center_y, width, height]
                    }
                }

                detections.append(detection)
            
            # JSONファイルに保存
            json_output_path = os.path.join(result_boxes_dir_path, f"{image_name}.json")
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(detections, f, indent=4, ensure_ascii=False)
