"""Sam's tools

Extra functions to use in conjunction with the yolox server 

Easy Usage: Import this module and call detection()
"""

import cv2
from requests import post
import numpy as np
import json

def start(frame):
    ret, im_np = cv2.imencode('.jpg', frame)
    im_byte = im_np.tobytes()
    dets = get_dets(im_byte)
    return dets

def get_dets(im_byte):
    metadata = post('http://127.0.0.1:5001/predict', data=im_byte)
    # print(metadata.text)
    return metadata.json()

def format_dets(raw_dets, thres):
    dets2 = []
    # Skip some nonsense frame
    if len(raw_dets) > 4:
        return dets2
    boxes, scores, cls_ids, class_names = raw_dets
    boxes = boxes.astype('int16').tolist()
    cls_ids = cls_ids.astype('int16').tolist()
    scores = scores.astype('float32').tolist()
    # print(f"boxes:{boxes}")
    # print(f"scores:{scores}")
    # print(f"cls_ids:{cls_ids}")
    # print(f"class_names:{class_names}")

    for i in range(len(boxes)):
        box = boxes[i]
        # print(f"box:{box}")
        cls_id = int(cls_ids[i])
        # print(f"cls_id:{cls_id}")
        # if yoloClass is not None and cls_id not in yoloClass:
        #     return None
        if scores[i] < thres:
            print(scores[i])
            print(thres)
            continue
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        w = int(x2 - x1)
        x = int(x1 + w / 2)
        h = int(y2 - y1)
        y = int(y1 + w / 2)

        # THESE CONDITIONS ARE FOR BYTETRACK SERVER ONLY
        aspect_ratio_thresh=1.6
        min_box_area=10
        vertical = w/h > aspect_ratio_thresh
        if w*h > min_box_area and not vertical:
            det = {}
            det['xyxy'] = [int(x1),int(y1),int(x2),int(y2)]
            det['xywh'] = [x,y,w,h]
            det['conf'] = scores[i]
            det['cls'] = class_names[cls_id]
            dets2.append(det)

    # print(dets2)
    return dets2

def detection(frame):
    dets = start(frame)
    # print(f"DETS, {dets}")
    return dets


