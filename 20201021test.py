import numpy as np
import matplotlib.pyplot as plt
import math

w = 3
b = 0.5

x_lin = np.linspace(0, 100, 101)
y = (x_lin + np.random.randn(101) * 5) * w + b
y_hat = x_lin * w + b

plt.plot(x_lin, y, 'b.', label = 'data')
plt.plot(x_lin, y_hat, 'r-', label = 'prediction')
plt.title("Assume we have data points (And the prediction)")
plt.legend(loc = 2)

def mean_absolute_error(y, yp):
    """
    計算 MAE
    Args:
        - y: 實際值
        - yp: 預測值
    Return:
        - mae: MAE
    """

    #math.pow(5,2)平方
    mae = sum(abs(y - yp)) / len(y)
    return mae

MAE = mean_absolute_error(y, y_hat)
print("The Mean absolute error is %.3f" % (MAE))

def mean_square_error(y, yp):
    mse = sum(np.power((y - yp), 2)) / len(y)
    return mse
mse = mean_square_error(y, y_hat)
print("The Mean square error is %.3f" % (mse))

plt.show()