import cv2
import numpy as np
import os
from datetime import datetime

# Create directories to store images
if not os.path.exists('data/up_move'):
    os.makedirs('data/up_move')
if not os.path.exists('data/down_move'):
    os.makedirs('data/down_move')

# Capturing the video file 0 for videocam else you can provide the url
video_path = "/Users/admin/Downloads/test2.mp4"
capture = cv2.VideoCapture(video_path)

# Reading the first frame
_, frame1 = capture.read()
# Convert to gray scale
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
# Create mask
hsv_mask = np.zeros_like(frame1)
# Make image saturation to a maximum value
hsv_mask[..., 1] = 255

# Flag for pause/resume
pause = False

while True:
    if not pause:
        _, frame2 = capture.read()
        if frame2 is None:
            break

        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Optical flow is now calculated
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Compute magnitude and angle of 2D vector
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Set image hue value according to the angle of optical flow
        hsv_mask[..., 0] = ang * 180 / np.pi / 2
        # Set value as per the normalized magnitude of optical flow
        hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # Convert to rgb
        rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

        combined_frame = np.hstack((frame2, rgb_representation))
        cv2.imshow('Original vs Optical Flow', combined_frame)

    kk = cv2.waitKey(30) & 0xFF

    # Press 'e' to exit the video
    if kk == ord('q'):
        break
    # Press 'p' to pause/resume
    elif kk == ord('p'):
        pause = not pause
    # Press up arrow to save current frame to directory data/up_move
    elif kk == ord('u'):  # Up arrow key
        filename = f"data/up_move/{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        cv2.imwrite(filename, rgb_representation)
        print(f"Saved frame as {filename}")
    # Press down arrow to save current frame to directory data/down_move
    elif kk == ord('d'):  # Down arrow key
        filename = f"data/down_move/{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        cv2.imwrite(filename, rgb_representation)
        print(f"Saved frame as {filename}")

    prvs = next

capture.release()
cv2.destroyAllWindows()
