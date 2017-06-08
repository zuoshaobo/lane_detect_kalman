from __future__ import division

import cv2
import numpy as np
import copy 


class LaneDetector:
    def __init__(self, road_horizon, prob_hough=True):
        self.prob_hough = prob_hough
        self.vote = 80
        self.roi_theta = 0.45
        self.road_horizon = road_horizon

    def _standard_hough(self, img, init_vote):
        # Hough transform wrapper to return a list of points like PHough does
        lines = cv2.HoughLines(img, 1, np.pi/180, init_vote)
        points = [[]]
        for l in lines:
            for rho, theta in l:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*a)
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*a)
                points[0].append((x1, y1, x2, y2))
        return points

    def _base_distance(self, x1, y1, x2, y2, width):
        # compute the point where the give line crosses the base of the frame
        # return distance of that point from center of the frame
        if x2 == x1:
            return (width*0.5) - x1
        m = (y2-y1)/(x2-x1)
        c = y1 - m*x1
        base_cross = -c/m
        return (width*0.5) - base_cross

    def _scale_line(self, x1, y1, x2, y2, frame_height):
        # scale the farthest point of the segment to be on the drawing horizon
        if y1 < y2:
            m = (y1-y2)/(x1-x2)
            x1 = ((0.67*frame_height-y1)/m) + x1
            y1 = 0.67*frame_height
            x2 = ((frame_height-y2)/m) + x2
            y2 = frame_height
        else:
            m = (y2-y1)/(x2-x1)
            x2 = ((0.67*frame_height-y2)/m) + x2
            y2 = 0.67*frame_height
            x1 = ((frame_height-y1)/m) + x1
            y1 = frame_height
        return x1, y1, x2, y2

    def detect(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        roiy_end = frame.shape[0]
        roix_end = frame.shape[1]
        roi = img[self.road_horizon:roiy_end, 0:roix_end]

	roi = cv2.GaussianBlur(roi,(5,5),0)


	hls=cv2.cvtColor(frame,cv2.COLOR_RGB2HLS).astype(np.float)
	hsv=cv2.cvtColor(frame,cv2.COLOR_RGB2HSV).astype(np.float)
	s_chanel=hls[:,:,2].astype(np.uint8)
	v_chanel=hsv[:,:,2].astype(np.uint8)
	ret,s_chanel=cv2.threshold(s_chanel,105,255,cv2.THRESH_BINARY) 
	#ret,v_chanel=cv2.threshold(v_chanel,200,255,cv2.THRESH_BINARY) 
	roi=s_chanel+v_chanel


	threshold,imgOtsu = cv2.threshold(roi,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	print threshold
        roi= cv2.Canny(roi, threshold/2, threshold)
        #roi= cv2.Canny(roi, 200, 550)
	print roi.dtype
	res=cv2.resize(roi,(350,300),interpolation=cv2.INTER_CUBIC)
	cv2.imshow('r2',res)
	roi = cv2.Sobel(roi,cv2.CV_8U,1,0,ksize=5)
	contours=roi
	ret,contours=cv2.threshold(contours,150,255,cv2.THRESH_BINARY) 


	'''
	kernel = np.ones((2,2),np.uint8)
	contours=cv2.erode(contours,kernel,iterations = 3)
	contours= cv2.dilate(contours,kernel,iterations = 4) 

	kernel = np.ones((3,3),np.uint8)
	kernel2 = np.ones((1,1),np.uint8)
	contours=cv2.erode(contours,kernel,iterations = 1)
	res=cv2.resize(contours,(350,300),interpolation=cv2.INTER_CUBIC)
	cv2.imshow('r3',res)
	'''

	#hls = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS).astype(np.float)
	#s_channel = hls[:,:,2].astype(np.uint8)
        #contours = cv2.Canny(s_channel, 60, 120)

	imshape=frame.shape
	print imshape
    	vertices = np.array([[(.55*imshape[1], 0.67*imshape[0]), (imshape[1],imshape[0]),
                        (0,imshape[0]),(.45*imshape[1], 0.67*imshape[0])]], dtype=np.int32)
        mask = np.zeros_like(contours, dtype=np.uint8)
        cv2.fillPoly(mask, vertices, 255)
        masked_image = cv2.bitwise_and(contours, mask)
	res=cv2.resize(contours,(350,300),interpolation=cv2.INTER_CUBIC)
	cv2.imshow('r',res)
	#cv2.waitKey(0)
	contours=masked_image


        if self.prob_hough:
            lines = cv2.HoughLinesP(contours, 1, np.pi/180, self.vote, minLineLength=30, maxLineGap=100)
        else:
            lines = self.standard_hough(contours, self.vote)
	
	if lines == None:
		print "None"
		return [None,None],frame
	#'''
        lines2 = lines+np.array([0, self.road_horizon, 0, self.road_horizon]).reshape((1, 1, 4))  # scale points from ROI coordinates to full frame coordinates
	frame2=copy.deepcopy(frame)
	for t in lines2:
            for x1,y1,x2,y2 in t:
                    theta = np.abs(np.arctan2((y2-y1), (x2-x1)))  # line angle WRT horizon
                    if theta > self.roi_theta:  # ignore lines with a small angle WRT horizon
	  		cv2.line(frame2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
	res=cv2.resize(frame2,(350,300),interpolation=cv2.INTER_CUBIC)
        cv2.imshow('q', res)
	#'''

        if lines is not None:
            # find nearest lines to center
            lines = lines+np.array([0, self.road_horizon, 0, self.road_horizon]).reshape((1, 1, 4))  # scale points from ROI coordinates to full frame coordinates

            left_bound = None
            right_bound = None
            for l in lines:
                # find the rightmost line of the left half of the frame and the leftmost line of the right half
                for x1, y1, x2, y2 in l:
                    theta = np.abs(np.arctan2((y2-y1), (x2-x1)))  # line angle WRT horizon
                    if theta > self.roi_theta:  # ignore lines with a small angle WRT horizon
                        dist = self._base_distance(x1, y1, x2, y2, frame.shape[1])
                        if left_bound is None and dist < 0:
                            left_bound = (x1, y1, x2, y2)
                            left_dist = dist
                        elif right_bound is None and dist > 0:
                            right_bound = (x1, y1, x2, y2)
                            right_dist = dist
                        elif left_bound is not None and 0 > dist > left_dist:
                            left_bound = (x1, y1, x2, y2)
                            left_dist = dist
                        elif right_bound is not None and 0 < dist < right_dist:
                            right_bound = (x1, y1, x2, y2)
                            right_dist = dist
            if left_bound is not None:
	    	centerx=(left_bound[0]+left_bound[2])/2.0
	    	if centerx>frame.shape[1]/2.0:
		    left_bound=None
	    	else:
                    left_bound = self._scale_line(left_bound[0], left_bound[1], left_bound[2], left_bound[3], frame.shape[0])
            if right_bound is not None:
	    	centerx=(right_bound[0]+right_bound[2])/2.0
	    	if centerx<frame.shape[1]/2.0:
		    right_bound=None
	    	else:
               	    right_bound = self._scale_line(right_bound[0], right_bound[1], right_bound[2], right_bound[3], frame.shape[0])

            return [left_bound, right_bound],contours

