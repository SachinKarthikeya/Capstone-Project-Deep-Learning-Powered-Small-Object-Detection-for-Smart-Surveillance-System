import cv2

class Camera:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)

    def get_frame(self):
        success, frame = self.cap.read()
        if not success:
            return None
        return frame
