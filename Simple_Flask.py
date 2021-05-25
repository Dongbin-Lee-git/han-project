import numpy as np
import cv2
import tensorflow as tf
from pose_estimation.applications import cam
from obj_detect.object_detect import obj_detect
from flask import Flask, request, Response
from hand_tracking.detect_single_threaded import hand_track
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# API
app = Flask(__name__)

# route http post to this method
@app.route('/api/upload', methods=['POST'])
def upload():
    # retrieve image from client
    img = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # process image
    hand_track(img)
    cam.run(img)
    obj_detect(img)

    # response


# start server
app.run(host="0.0.0.0", port=5000)
