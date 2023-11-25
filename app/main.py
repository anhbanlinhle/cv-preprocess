import cv2
import numpy as np
import os
from datetime import datetime

up_move_dir = 'data/up_move'
down_move_dir = 'data/down_move'
no_move_dir = 'data/no_move'

if not os.path.exists(up_move_dir):
    os.makedirs(up_move_dir)
if not os.path.exists(down_move_dir):
    os.makedirs(down_move_dir)
if not os.path.exists(no_move_dir):
    os.makedirs(no_move_dir)

down_move_files = os.listdir(down_move_dir)
no_move_files = os.listdir(no_move_dir)
up_move_files = os.listdir(up_move_dir)

up_move_index = len(up_move_files)
no_move_index = len(no_move_files)
down_move_index = len(down_move_files)

video_path = "test/test8.mp4"
capture = cv2.VideoCapture(video_path)

_, frame1 = capture.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv_mask = np.zeros_like(frame1)
hsv_mask[..., 1] = 255

pause = False

while True:
    if not pause:
        _, frame2 = capture.read()
        if frame2 is None:
            break

        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv_mask[..., 0] = ang * 180 / np.pi / 2
        hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

        combined_frame = np.hstack((frame2, rgb_representation))
        cv2.imshow('Original vs Optical Flow', combined_frame)

    kk = cv2.waitKey(30) & 0xFF

    if kk == ord('q'):
        break
    elif kk == ord('p'):
        pause = not pause
    elif kk == ord('u'):
        filename = f"data/up_move/up_move_" + str(up_move_index) + ".png"
        cv2.imwrite(filename, rgb_representation)
        up_move_index += 1

    elif kk == ord('d'):
        filename = "data/down_move/down_move_" + str(down_move_index) + ".png"
        cv2.imwrite(filename, rgb_representation)
        down_move_index += 1
        
    elif kk == ord('n'):
        filename = "data/no_move/no_move_" + str(no_move_index) + ".png"
        cv2.imwrite(filename, rgb_representation)
        no_move_index += 1


    prvs = next

capture.release()
cv2.destroyAllWindows()
