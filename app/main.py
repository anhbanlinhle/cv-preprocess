import cv2
import numpy as np
import os
from datetime import datetime

if not os.path.exists('data/up_move'):
    os.makedirs('data/up_move')
if not os.path.exists('data/down_move'):
    os.makedirs('data/down_move')
if not os.path.exists('data/no_move'):
    os.makedirs('data/no_move')

video_path = "test/test0.mp4"
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
        filename = f"data/up_move/{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        cv2.imwrite(filename, rgb_representation)
        print(f"Saved frame as {filename}")
    elif kk == ord('d'):
        filename = f"data/down_move/{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        cv2.imwrite(filename, rgb_representation)
        print(f"Saved frame as {filename}")
    elif kk == ord('n'):
        filename = f"data/no_move/{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        cv2.imwrite(filename, rgb_representation)
        print(f"Saved frame as {filename}")


    prvs = next

capture.release()
cv2.destroyAllWindows()
