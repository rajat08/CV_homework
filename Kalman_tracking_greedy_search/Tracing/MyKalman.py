import numpy as np


class MyKalman(object):
    def __init__(self):
        self.state = np.zeros((2, 1))  # previous state vector  #x[t-1]
        self.dt = 0.005  # delta time
        self.F = np.array([[1.0, self.dt], [0.0, 1.0]])
        self.P = np.diag((3.0, 3.0))  # covariance matrix   #P = M
        self.H = np.array([[1, 0], [0, 1]])  #Observation Matrix      #H
        self.observations = np.array([[0], [255]])  # vector of observations #centroid
        self.Q = np.eye(self.state.shape[0])          #Q
        self.R = np.eye(self.observations.shape[0])         #R
        self.lastResult = np.array([[0], [0]])

    def predict(self):
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        self.lastResult = self.state
        return self.state

    def update(self, detection, flag):
        if not flag:  # update using prediction
            self.observations = self.lastResult
        else:  # update using detection
            self.observations = detection
        k = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(np.dot(self.H, np.dot(self.P, self.H.T)) + self.R)))
        z = self.observations + np.random.rand(2,1)
        self.state = self.state + np.dot(k, (z - np.dot(self.H, self.state)))
        self.P = self.P - np.dot(k, np.dot(self.H, self.P))
        self.lastResult = self.state
        return self.state