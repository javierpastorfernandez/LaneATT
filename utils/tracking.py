import cv2
import numpy as np
from scipy.linalg import block_diag
from utils.openlane_utils import bcolors,FormatAxes, get_cmap, create_trace_loglevel
import logging

class LaneTracker:
    def __init__(self, n_lanes, proc_noise_scale, meas_noise_scale,
                 process_cov_parallel=0, proc_noise_type='white'):

        self.logger = logging.getLogger(__name__)


        self.n_lanes = n_lanes
        self.meas_size = 4 * self.n_lanes # x11 vx11 y11 vy11 x12 vx12 y12 vy12 x13 vx13 y13 vy13
        self.state_size = self.meas_size * 2
        self.contr_size = 0

        self.kf = cv2.KalmanFilter(self.state_size, self.meas_size,
                                   self.contr_size)
        self.kf.transitionMatrix = np.eye(self.state_size, dtype=np.float32)
        self.kf.measurementMatrix = np.zeros((self.meas_size, self.state_size),
                                             np.float32)
        for i in range(self.meas_size):
            self.kf.measurementMatrix[i, i*2] = 1

        if proc_noise_type == 'white':
            block = np.matrix([[0.25, 0.5],
                               [0.5, 1.]], dtype=np.float32)
            proc_noise = block_diag(*([block] * self.meas_size))
        if proc_noise_type == 'identity':
            proc_noise = np.eye(self.state_size, dtype=np.float32)
        self.kf.processNoiseCov = proc_noise * proc_noise_scale

        # for i in range(0, self.meas_size, 2):
        #     for j in range(1, self.n_lanes):
        #         self.kf.processNoiseCov[i, i+(j*8)] = process_cov_parallel
        #         self.kf.processNoiseCov[i+(j*8), i] = process_cov_parallel

        self.kf.measurementNoiseCov = np.eye(self.meas_size, dtype=np.float32)\
                                      * meas_noise_scale

        self.kf.errorCovPre = np.eye(self.state_size)

        self.meas = np.zeros((self.meas_size, 1), np.float32)
        self.state = np.zeros((self.state_size, 1), np.float32)

        self.first_detected = False

    def _update_dt(self, dt):
        for i in range(0, self.state_size, 2):
            self.kf.transitionMatrix[i, i+1] = dt

    def _first_detect(self, lanes):
        for l, i in zip(lanes, range(0, self.state_size, 8)):
            self.state[i:i+8:2, 0] = l
        self.kf.statePost = self.state
        self.first_detected = True

    def update(self, lanes):
        self.logger.trace(bcolors.OKGREEN+'lanes: ' +bcolors.ENDC+str(lanes))
        self.first_detected=True


        if self.first_detected:
            if lanes:
                idx=0
                for lane in lanes:
                    for coords in lane: # punto inicial, final
                        self.meas[idx, 0] = coords[0]# x
                        idx+=1
                        self.meas[idx, 0] = coords[1]# y
                        idx+=1


            self.logger.trace(bcolors.OKGREEN+'self.meas: ' +bcolors.ENDC+str(self.meas))
            breakpoint()
            self.kf.correct(self.meas)
        else:
            if lanes.count(None) == 0:
                self._first_detect(lanes)


    def predict(self, dt):
        if self.first_detected:
            self._update_dt(dt)
            state = self.kf.predict()
            lanes = []
            for i in range(0, len(state), 8):
                lanes.append((state[i], state[i+2], state[i+4], state[i+6]))
            return lanes
        else:
            return None