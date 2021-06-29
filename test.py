import numpy as np
import cv2

cap = cv2.VideoCapture('test.mp4')

# Define the codec and create VideoWriter object
#fourcc = cv2.cv.CV_FOURCC(*'DIVX')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
fourcc = cv2.VideoWriter_fourcc(*'XVID')


out = cv2.VideoWriter('output.avi', fourcc, 20.0,
                      (int(cap.get(3)), int(cap.get(4))), isColor=False)
backSub = cv2.createBackgroundSubtractorKNN()
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame, 0)
        fgMask = backSub.apply(frame)
        # write the flipped frame
        out.write(fgMask)

        cv2.imshow('frame', fgMask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
