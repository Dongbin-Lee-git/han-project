import numpy as np
import cv2 as cv

print(cv.__file__)

net = cv.dnn.readNet('yolov3-custom_last.weights', 'yolov3-custom.cfg')
inp = np.random.standard_normal([1, 3, 608, 608]).astype(np.float32)
net.setInput(inp)
out = net.forward()
print(out.shape)