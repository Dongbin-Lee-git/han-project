import cv2
import tensorflow as tf
import visualizations as vis
from pose_estimation.applications.model_wrapper import ModelWrapper
import pose_estimation.configs.draw_config as draw_config
model_path = "../trained_models/model11_test-15Sun1219-2101"
model_wrapper = ModelWrapper(model_path)

def run(image):
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    skeletons = model_wrapper.process_image(img_rgb)
    skeleton_drawer = vis.SkeletonDrawer(img_rgb, draw_config)
    for skeleton in skeletons:
        skeleton.draw_skeleton(skeleton_drawer.joint_draw, skeleton_drawer.kpt_draw)
    processed_img_rgb = img_rgb
    processed_img_bgr = cv2.cvtColor(processed_img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite("processed_pose.jpg", processed_img_bgr)

run("testimg.jpg")
