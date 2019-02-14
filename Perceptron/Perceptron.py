import numpy as np
#eta is learning rate, n_iter is the number of updates
class Perceptron(object):
    def __init__(self,eta = .01, n_iter = 50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    #Fits training data
    #X = [n_samples, n_features]
    #Y is [n_samples] <--- Target values
    def fit(self,X,Y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = .01, size = 1 + X.shape[1]) #initializes your weights by sampling a normal distribution n + 1 times
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,Y): #zip(X,Y) looks like [[features, class]]
                delta_w = self.eta * (target - self.predict(xi))
                self.w_[1:] += delta_w * xi
                self.w_[0] += delta_w #updates bias
                errors += int(delta_w != 0.0)
            self.errors_.append(errors)
        return self
    def net_input(self,x):
        return np.dot(x,self.w_[1:]) + self.w_[0]

    def predict(self, x):
        #return class label
        return np.where(self.net_input(x) >= 0.0 , 1, -1)
