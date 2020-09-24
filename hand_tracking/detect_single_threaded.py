from hand_tracking.utils import detector_utils as detector_utils
import cv2
import uuid
import json

detection_graph, sess = detector_utils.load_inference_graph()

def hand_track(image_np):
    # image_np = cv2.imread(image, cv2.IMREAD_COLOR)
    h, w, c = image_np.shape
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sess)
    detector_utils.draw_box_on_image(2, 0.2, scores, boxes, w, h, image_np)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    #save file
    path_file = ('static/%s.jpg' %uuid.uuid4().hex)
    cv2.imwrite(path_file, image_np)

    return json.dumps(path_file) #return image file name

