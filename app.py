from flask import Flask, request
from werkzeug.utils import secure_filename
import cv2 as cv
import numpy as np
app = Flask(__name__)


@app.route('/')
def home():
    return "Hello World"


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        img2bgsub()
        return 'file uploaded successfully'


def img2bgsub():
    cap = cv.VideoCapture('test.mp4')
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('output.avi', fourcc, 20.0,
                         (int(cap.get(3)), int(cap.get(4))), isColor=False)

    backSub = cv.createBackgroundSubtractorKNN()
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            #frame = cv.flip(frame, 0)
            fgMask = backSub.apply(frame)

            out.write(fgMask)

            #cv.imshow('frame', frame)

        else:
            break
    cap.release()
    out.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    app.run(debug=True)
