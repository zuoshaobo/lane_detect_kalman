import cv2.cv as cv2
import numpy as np
from scipy.linalg import block_diag


class LaneTracker:
    def __init__(self, n_lanes, proc_noise_scale, meas_noise_scale, process_cov_parallel=0, proc_noise_type='white'):
        self.n_lanes = n_lanes
        self.meas_size = 4 * self.n_lanes
        self.state_size = self.meas_size * 2
        self.contr_size = 0

	self.kalman=cv2.CreateKalman(self.state_size,self.meas_size,self.contr_size)

	#print self.kalman.transition_matrix.cols
	#print self.kalman.transition_matrix.rows
	#print self.kalman.transition_matrix.step
	t2=np.eye(self.state_size, dtype=np.float32)
	for t in range(self.kalman.transition_matrix.rows):
		for tt in range(self.kalman.transition_matrix.cols):
			self.kalman.transition_matrix[t,tt]=t2[t,tt]


	#print self.kalman.measurement_matrix
	for t in range(self.kalman.measurement_matrix.rows):
		for tt in range(self.kalman.measurement_matrix.cols):
			self.kalman.measurement_matrix[t,tt]=0
        for i in range(self.meas_size):
            self.kalman.measurement_matrix[i, i*2] = 1

        if proc_noise_type == 'white':
            block = np.matrix([[0.25, 0.5],
                               [0.5, 1.]], dtype=np.float32)
            t2= (block_diag(*([block] * self.meas_size)) * proc_noise_scale)
	for t in range(self.kalman.process_noise_cov.rows):
		for tt in range(self.kalman.process_noise_cov.cols):
			self.kalman.process_noise_cov[t,tt]=t2[t,tt]
			#print self.kalman.process_noise_cov[t,tt]

	'''
        if proc_noise_type == 'identity':
            self.kalman.processNoiseCov = np.eye(self.state_size, dtype=np.float32) * proc_noise_scale
	'''
        for i in range(0, self.meas_size, 2):
            for j in range(1, self.n_lanes):
                self.kalman.process_noise_cov[i, i+(j*8)] = process_cov_parallel
                self.kalman.process_noise_cov[i+(j*8), i] = process_cov_parallel
	



	#print self.kalman.measurement_noise_cov
        t2=np.eye(self.meas_size, dtype=np.float32) *meas_noise_scale
	for t in range(self.kalman.measurement_noise_cov.rows):
		for tt in range(self.kalman.measurement_noise_cov.cols):
			self.kalman.measurement_noise_cov[t,tt]=t2[t,tt]

	#print self.kalman.error_cov_pre
        t2=np.eye(self.state_size, dtype=np.float32) *meas_noise_scale
	for t in range(self.kalman.error_cov_pre.rows):
		for tt in range(self.kalman.error_cov_pre.cols):
			self.kalman.error_cov_pre[t,tt]=t2[t,tt]



	self.state=cv2.CreateMat(self.state_size, 1, cv2.CV_32FC1)
	self.meas=cv2.CreateMat(self.meas_size, 1, cv2.CV_32FC1)

        self.first_detected = False

    def _update_dt(self, dt):
        for i in range(0, self.state_size, 2):
            self.kalman.transition_matrix[i, i+1] = dt

    def _first_detect(self, lanes):
        for l, i in zip(lanes, range(0, self.state_size, 8)):
		if l is not None:
			c=0
			for t in range(i,i+8,2):
				self.kalman.state_post[t,0]=l[c]
				c+=1
        self.first_detected = True

    def update(self, lanes):
        if self.first_detected:
            for l, i in zip(lanes, range(0, self.meas_size, 4)):
		if l is not  None:
			c=0
			for t in range(i,i+4):
				self.meas[t,0]=l[c]
				c+=1
	    cv2.KalmanCorrect(self.kalman,self.meas)
        else:
            if lanes.count(None) == 0:
		print "============"
                self._first_detect(lanes)

    def predict(self, dt):
        if self.first_detected:
            self._update_dt(dt)
            state = cv2.KalmanPredict(self.kalman)
            lanes = []
            for i in range(0, state.rows, 8):
                lanes.append((state[i,0], state[i+2,0], state[i+4,0], state[i+6,0]))
		#print 'r:',state[i,0],state[i+2,0],state[i+4,0],state[i+6,0]
            return lanes
        else:
            return None



