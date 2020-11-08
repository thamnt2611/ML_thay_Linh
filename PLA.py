import numpy as np
import matplotlib.pyplot as plt
# np.random.seed(3)

def gradvec(x, y):
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    grad = -np.dot(x.T, y)
    return grad/np.linalg.norm(grad)

class PLA():
    def __init__(self, weight=None):
        self.weight = weight

    def fit(self, weight_init, X, y, eta = 0.01, epochs = 10000):
        self.weight = weight_init
        cycle = 10
        count = 0
        w_start_cycle = self.weight

        for x in range(epochs):
            index = np.random.permutation(X.shape[0])
            for i in index:
                x_i = X[i, :]
                y_i = y[i]
                gradient = gradvec(x_i, y_i)
                w_old = self.weight
                self.weight = w_old - eta*gradient
                count += 1
                if(count==cycle):
                    count = 0
                    y_predict = self.predict(X, self.weight)
                    if (np.linalg.norm(self.weight - w_start_cycle)< 1e-4):
                        print(x)
                        break
                    w_start_cycle = self.weight
        return self.weight

    def predict(self, X, weight):
        res = np.dot(X, weight)
        y = []
        for i in res:
            print(i)
            if i>=0 :
                y.append([1])
            else:
                y.append([0])
        return y

    # def score(self, X, y):
    #     y_predict = self.predict(X)
    #     true_cnt = 0
    #     total = y.shape[0]
    #     for i in range(total):
    #         if y[i]==y_predict[i]:
    #             true_cnt+=1
    #     return true_cnt/total

if __name__=='__main__':
    means = [[2,2], [6, 2]]
    cov = [[.3, .1], [.1, .3]]
    N=100
    X0 = np.random.multivariate_normal(means[0], cov, N)
    X1 = np.random.multivariate_normal(means[1], cov, N)
    X = np.concatenate((X0, X1), axis = 0)
    y = np.concatenate((np.ones((N, 1)), -1*np.ones((N, 1))), axis=0)
    X = np.concatenate((X, np.ones((2*N, 1))), axis = 1)
    # print(X)
    # print(y)
    pla = PLA()
    weight_init = np.zeros((X.shape[1], 1))
    print(weight_init)
    pla.fit(weight_init,X, y)
    weight = pla.weight
    # print(weight)
    # print(pla.predict(X))
    plt.figure(figsize = (5, 3))
    plt.scatter(X0[:,0], X0[:,1], color = 'red')
    plt.scatter(X1[:,0], X1[:,1], color = 'blue')
    plt.scatter(means[0][0], means[0][1], s = 40, color ='yellow')
    plt.scatter(means[1][0], means[1][1], s = 40, color ='green')
    x_axis = np.array([0, 5])
    plt.plot(x_axis, -(weight[0]*x_axis + weight[2])/weight[1])
    plt.plot(x_axis, -(weight_init[0]*x_axis + weight_init[2])/weight_init[1], color='black')
    plt.show()




