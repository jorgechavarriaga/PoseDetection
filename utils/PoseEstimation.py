# https://google.github.io/mediapipe/solutions/pose.html

import cv2
import mediapipe as mp
import utils.BGRColor as BGR

class poseDetector():
    def __init__(self, staticImage = False, modelComplex = 1, smoothLandmarks = True,  
                 smoothSegmentation = True, minDetectionConfidence = 0.75, 
                 minTrackingConfidence = 0.75):
        self.staticImage = staticImage
        self.modelComplex = modelComplex
        self.smoothLandmarks = smoothLandmarks
        self.smoothSegmentation = smoothSegmentation
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence
        
        self.mpDrawing = mp.solutions.drawing_utils
        self.mpDrawingStyles = mp.solutions.drawing_styles
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.staticImage, self.modelComplex, self.smoothLandmarks, 
                self. smoothSegmentation, self.minDetectionConfidence, self.minTrackingConfidence)

    def findPose(self, img, drawing = True):
        imgToRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgToRGB)
        if self.results.pose_landmarks:
            if drawing:
                # print(results.pose_landmarks)
                self.mpDrawing.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS, 
                                landmark_drawing_spec=self.mpDrawingStyles.get_default_pose_landmarks_style())
        return img
    
    def findPosition(self, img, drawing = True):
        landmarkList = []
        if self.results.pose_landmarks:
            for keypoint, landmark in enumerate(self.results.pose_landmarks.landmark):
                height, width, _ = img.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                landmarkList.append([keypoint, cx,cy])
                # print(keypoint, cx, cy)
                if drawing:
                    cv2.circle(img, (cx, cy), 5, BGR.GREEN, cv2.FILLED)
        return landmarkList


