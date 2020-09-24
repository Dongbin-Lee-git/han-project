import cv2

class SkeletonDrawer:
    def __init__(self, img, draw_config):
        self.img = img
        self.dc = draw_config

    def change_color(self, joint):
        self.dc.joint_colors_bgr[joint] = (self.dc.joint_colors_bgr[joint][1], self.dc.joint_colors_bgr[joint][0], 0)

    def _scale_flip_coord(self, coord):
        y = coord[0]
        x = coord[1]
        scaled_y = int(y * self.img.shape[0])
        scaled_x = int(x * self.img.shape[1])
        return scaled_x, scaled_y

    def joint_draw(self, start_coord, end_coord, joint_name):
        start_coord = self._scale_flip_coord(start_coord)
        end_coord = self._scale_flip_coord(end_coord)
        color = self.dc.joint_colors_bgr[joint_name]
        cv2.line(self.img, start_coord, end_coord, color, self.dc.joint_line_thickness, lineType=cv2.LINE_AA)
        print(start_coord, end_coord, joint_name)

    def kpt_draw(self, kpt_coord, kpt_name):
        kpt_coord = self._scale_flip_coord(kpt_coord)
        cv2.circle(self.img, kpt_coord, self.dc.keypoint_circle_diameter, self.dc.keypoint_circle_color)
