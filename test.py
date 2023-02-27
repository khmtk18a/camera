import cv2
from camera import Camera, cropped_frame
from cv2 import Mat

cw = ch = 300
with Camera() as video:
    # https://www.geeksforgeeks.org/python-background-subtraction-using-opencv/
    bs = cv2.createBackgroundSubtractorKNN(0, 50)
    while True:
        ret: bool; frame: Mat

        ret, frame = video.read()

        if not ret:
            print("Something is not working.")
            break

        x = frame.shape[1] - cw
        y = 0

        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (x, y), (x+cw, y+ch), (255, 0, 0))

        cv2.imshow('camera', frame)

        fgmask = bs.apply(frame)
        fgmask = cropped_frame(fgmask, x, y, cw, ch)

        cv2.imshow('cropped', fgmask)

        # press 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
