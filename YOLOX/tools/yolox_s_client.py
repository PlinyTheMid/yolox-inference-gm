import cv2
import time
import ast
import numpy as np

from tools import detection
from _collections import deque
import matplotlib.pyplot as plt
from tracking_Sam.tracker.byte_tracker import BYTETracker

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


FRAME_SIZE = (1280,720)
# input = "/home/ubuntu/ByteTrack-test-videos/Hyd_video.mp4"
input = "/home/ubuntu/Client-Videos/[Natalie]_ANPR_trucks/B16.mp4"
# input = "/home/ubuntu/Client-Videos/Tracking_test_videos/aod_video_5.mp4"
cap = cv2.VideoCapture(input)

if cap.isOpened() == False:
    print("Error in input video")

out = cv2.VideoWriter("test.avi", cv2.VideoWriter_fourcc(*'XVID'), 10, FRAME_SIZE)

# Initialize ByteTrack
byteMe = BYTETracker()
pts = [deque(maxlen=30) for _ in range(1000)]

while True:
    ret, raw = cap.read()
    if ret == False:
        break
    frame = raw.copy()
    # print(f'frame.shape before resizing: {frame.shape}')
    frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_LINEAR)
    # print(f'frame.shape after resizing: {frame.shape}')
    # time.sleep(5)
    dets = detection(frame)
    # print(dets) # all detections for a single frame

    '''detections'''
    for i in range(len(dets)):
        det = dets[i]
        cls_name = det['cls']
        print(f"cls_name: {cls_name}")
        conf_det = det['conf']
        print(f"conf_det: {conf_det}")
        x1,y1,x2,y2 = det['xyxy']

        point1 = (x1,y1)
        point2 = (x2,y2)

        # if conf_det>0.25 and (cls_name == 'bus' or cls_name == 'truck' or cls_name == 'car'):
        cv2.rectangle(frame, point1, point2, (0, 0, 255), 2)
        cv2.putText(frame, f"{det['cls']}", point1, cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

    '''tracking'''
    # tracks = byteMe.update(dets)
    # for track in tracks:
    #     # print(f'track.cls: {track.cls}')
    #     # print(f'track.score: {track.score}')
    #     if track.attr['conf'] < 0.1: #0.5
    #         print(f"track_conf: {track.attr['conf']}")
    #         continue        
    #     x1,y1,x2,y2 = track.attr['xyxy']
    #     point1 = (x1,y1)
    #     point2 = (x2,y2)
    #     '''Invisible Tracking - Tracked objects are of same color'''
    #     # if track.score>0.25 and (track.cls == 'bus' or track.cls == 'truck' or track.cls == 'car'):
    #     #     print(f'track.cls: {track.cls}')
    #     #     cv2.rectangle(frame, point1, point2, (0, 255, 0), 2)
    #     #     cv2.putText(frame, f"{track.attr['cls']}", point1, cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    #     '''Method 1: Colored Boxes'''
    #     intbox = tuple(map(int, (x1, y1, x1 + x2, y1 + y2)))
    #     text_scale = max(1, frame.shape[1] / 1600.)
    #     text_thickness = 1
    #     line_thickness = max(1, int(frame.shape[1] / 500.))

    #     obj_id = int(track.id)
    #     id_text = '{}'.format(int(obj_id))
    #     # bbox_xy = f'w:{x2-x1} \n h:{y2-y1}'

    #     color = get_color(abs(obj_id))
    #     # 4W Tracking
    #     if track.score>0.25 and (track.cls == 'bus' or track.cls == 'truck' or track.cls == 'car'):
    #     # Bag Tracking
    #     # if track.score>0.1 and (track.cls == 'handbag' or track.cls == 'backpack' or track.cls == 'suitcase'):
    #         print(f'track.cls: {track.cls}')
    #         cv2.rectangle(frame, point1, point2, color=color, thickness=line_thickness)
    #         cv2.putText(frame, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (255, 255, 255),
    #                     thickness=text_thickness)
    #         # cv2.putText(frame, bbox_xy, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 0),
    #                     # thickness=text_thickness)

    #     '''Method 2: Tracking trails'''
        # cmap = plt.get_cmap('tab20b')
        # colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]
        # color = colors[obj_id % len(colors)]
        # color = [i * 255 for i in color]
        # center =  (int(((x2+x1)/2)), int(y2))
        # pts[obj_id].append(center)
        # for j in range(1, len(pts[obj_id])):
        #     # print(f'j: {j}')
        #     if pts[obj_id][j-1] is None or pts[obj_id][j] is None:
        #         continue
        #     # thickness_trail = int(np.sqrt(64/float(j+1))*2)
        #     thickness_trail = int(10/(np.sqrt(32/float(j+1))*2.5))
        #     print(f'thickness_trail: {thickness_trail}')
        #     cv2.line(frame, (pts[obj_id][j-1]), (pts[obj_id][j]), (0, 0, 255), thickness_trail)

        # print(track.id)
    out.write(frame)

cap.release()
out.release()

