import cv2
import numpy as np
class VideoProcessor():
    def __init__(self, video):
        """
            Initialize the video with basic details about the video.
        """
        self.video = video
        cap = cv2.VideoCapture(video)
        self.fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count/self.fps
        cap.release()
        print(f"{self.fps} {self.frame_count} {self.duration}")
        self.frames = []
        
    def read_frames(self, forced=False):
        if not forced and len(self.frames) != 0:
            return
        self.frames = []
        cap = cv2.VideoCapture(self.video)
        while cap.grab():
            poss, img = cap.retrieve()
            if not poss:
                break
            img_np = np.array(img)
            frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            self.frames.append(frame)