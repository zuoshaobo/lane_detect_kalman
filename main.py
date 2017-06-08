from __future__ import division

import cv2
import numpy as np

import track
import detect
import time

#video_path='harder_challenge_video.mp4'
video_path='../Mexico_raining_on_Highway.mp4'
video_path='/home/pub/Desktop/New/2017_0606_090815_260.MOV'
video_path='project_video.mp4'
video_path='/home/pub/Desktop/New/2017_0607_193733_282.MOV'
video_path='/home/pub/Desktop/New/2017_0606_183543_265.MOV'
video_path='/home/pub/Desktop/New/2017_0606_182543_263.MOV'
video_path='/home/pub/Desktop/New/2017_0606_183043_264.MOV'
video_path='/home/pub/Desktop/201706050819_000033AA.MP4'
video_path='challenge_video.mp4'
video_path='/home/pub/Desktop/New/2017_0606_182543_263.MOV'
video_path='/home/pub/Desktop/New/2017_0607_194233_283.MOV'
def main(video_path):
    cap = cv2.VideoCapture(video_path)

    ticks = 0

    lt = track.LaneTracker(2, 0.1, 500)
    ld = detect.LaneDetector(0)
    while cap.isOpened():
        precTick = ticks
        ticks = cv2.getTickCount()
        dt = (ticks - precTick) / cv2.getTickFrequency()
	dt*=0.2

        ret, frame = cap.read()
	frame=cv2.resize(frame,(1280,720),interpolation=cv2.INTER_CUBIC)

	time1=time.time()
        predicted = lt.predict(dt)

        lanes ,frame2= ld.detect(frame)
	print time.time()-time1
	imshape=frame.shape
    	vertices = np.array([[(.55*imshape[1], 0.67*imshape[0]), (imshape[1],imshape[0]),
                        (0,imshape[0]),(.45*imshape[1], 0.67*imshape[0])]], dtype=np.int32)
	cv2.polylines(frame,[vertices],True,(0,255,0),3)

	predicted2=[]
	if lanes.count(None)==0:
		#print lanes
		predicted2.append(map(int,list(lanes[0])))
		predicted2.append(map(int,list(lanes[1])))
	else:
	    predicted2=None
	 
  	if predicted is not None:
	  cv2.line(frame, (int(predicted[0][0]), int(predicted[0][1])), (int(predicted[0][2]), int(predicted[0][3])), (0, 0, 255), 5)
	  cv2.line(frame, (int(predicted[1][0]), int(predicted[1][1])), (int(predicted[1][2]), int(predicted[1][3])), (0, 0, 255), 5)
	'''
  	if predicted2 is not None:
	  cv2.line(frame, (int(predicted2[0][0]), int(predicted2[0][1])), (int(predicted2[0][2]), int(predicted2[0][3])), (255, 0, 0), 5)
	  cv2.line(frame, (int(predicted2[1][0]), int(predicted2[1][1])), (int(predicted2[1][2]), int(predicted2[1][3])), (255, 0, 0), 5)
	'''


        lt.update(lanes)

        cv2.imshow('', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
main(video_path)
