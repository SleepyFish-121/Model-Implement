import seaborn as sns

sns.set_context("poster")

if __name__ == '__main__':
    from MyImplementOfClassicModel.LinearModel import *

    n = int(input("this is the test of linear model\nPlease input the number of rows\n> "))
    p = int(input("Please input the number of  unknown coefficients(including constant)\n> "))
    X = np.random.rand(n, p - 1)
    b = np.random.rand(p - 1, 1)
    b0 = np.random.rand(1)
    y = np.matmul(X, b) + b0
    model, result = LinearRegreesion().fit(X, y)
    X = np.random.rand(1, p - 1)
    y = np.matmul(X, b) + b0
