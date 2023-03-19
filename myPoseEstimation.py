import os, cv2, time, utils.PoseEstimation as pe, utils.BGRColor as BGR

os.system('cls')

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = pe.poseDetector()
    while cap.isOpened():
        _, img              = cap.read()
        img = detector.findPose(img)
        landmarkList = detector.findPosition(img)
        for keypoint, cx, cy  in landmarkList:
            if keypoint == 11 or keypoint == 12 :
                print(f'KeyPoint: {keypoint} - Coordinates: ({cx},{cy})')
                if keypoint == 11:
                    cv2.putText(img, f'{cx},{cy} ', (cx,cy), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, BGR.RED, 1) 
                else:
                    cv2.putText(img, f'{cx},{cy} ', (cx,cy), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, BGR.RED, 1) 
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20,20), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, BGR.RED, 1)
        cv2.imshow('Pose Estimation', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
        
if __name__ == '__main__':
    main()