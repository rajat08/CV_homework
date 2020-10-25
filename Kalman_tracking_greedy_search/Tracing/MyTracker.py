
# Import python libraries
import numpy as np
from MyKalman import MyKalman
#from kalman_filter import KalmanFilter
from scipy.optimize import linear_sum_assignment
import cv2 as cv


class Track(object):

    def __init__(self, prediction, trackIdCount):
        self.track_id = trackIdCount
        self.KF = MyKalman()
        #self.KF = cv.KalmanFilter()
        self.prediction = np.asarray(prediction)
        #self.KF.update(detection)
        self.skipped_frames = 0
        self.trace = []  # trace path


class Tracker(object):


    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length,
                 trackIdCount):

        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.trackIdCount = trackIdCount

    def Update(self, detections):

        # Create tracks if no tracks vector found
        if (len(self.tracks) == 0):
            for i in range(len(detections)):
                track = Track(detections[i], self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Loss
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                try:
                    diff = self.tracks[i].prediction - detections[j]
                    distance = np.sqrt(diff[0][0]*diff[0][0] + diff[1][0]*diff[1][0])
                    cost[i][j] = distance
                except:
                    pass


        cost = (0.5) * cost
        # Hungarian
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)          # Returns a row index which are the Tracks and col_index which are the Detections
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]             #Assign row_index to col_index i.e assigning Tracks to detections

        #print(assignment)

        # Unassigned track
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if (assignment[i] != -1):
                # check for cost distance threshold.
                if (cost[i][assignment[i]] > self.dist_thresh):
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                pass
            else:
                self.tracks[i].skipped_frames += 1

        # If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                del_tracks.append(i)
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in del_tracks:
                if id < len(self.tracks):
                    del self.tracks[id]
                    del assignment[id]
                

        # Now look for un_assigned detects              # WHen number of centroids are > Number of tracks
        un_assigned_detects = []
        for i in range(len(detections)):
                if i not in assignment:
                    un_assigned_detects.append(i)

        # Start new tracks
        if(len(un_assigned_detects) != 0):
            for i in range(len(un_assigned_detects)):
                track = Track(detections[un_assigned_detects[i]],self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)


        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()

            if(assignment[i] != -1):
                self.tracks[i].skipped_frames = 0
                self.tracks[i].prediction = self.tracks[i].KF.update(detections[assignment[i]], 1)
            else:
                self.tracks[i].prediction = self.tracks[i].KF.update(np.array([[0], [0]]), 0)

            if(len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) - self.max_trace_length):
                    del self.tracks[i].trace[j]

            self.tracks[i].trace.append(self.tracks[i].prediction)
            self.tracks[i].KF.lastResult = self.tracks[i].prediction
