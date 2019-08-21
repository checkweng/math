##一元微分
def numerical_grad_1d(f, x, delta=1e-6):
    f_l, f_r = f(x + delta), f(x - delta)
    grad = (f_l - f_r) / (2 * delta)
    return grad


import numpy as np

f = lambda x: x ** 2 + x ** (1/3) + np.sin(x * np.cos(x) ** 2)
numerical_grad_1d(f, 2.31)


import tensorflow as tf # tensorflow2.0

x = tf.Variable(2.31)
with tf.GradientTape() as tape:
    g = x ** 2 + x ** (1/3) + tf.sin(x * tf.cos(x) ** 2)
    grad = tape.gradient(g, x)
grad.numpy()



##多元

def numerical_grad_nd(f, x, delta=1e-6):
    grad = np.zeros_like(x)
    for idx in range(x.size):
        temp_value = x[idx]
        x[idx] = temp_value - delta
        f_l = f(x)

        x[idx] = temp_value + delta
        f_r = f(x)

        grad[idx] = (f_r - f_l) / (2 * delta)
        x[idx] = temp_value
    return grad


f = lambda x: x[0] ** 2 + x[1] ** 2 - 5 * np.sin(x[2] ** 2) * np.tan(x[0] * x[1] * x[2])
x = np.array([2., 2.35, np.pi])
numerical_grad_nd(f, x)


x = tf.Variable([2, 2.35, np.pi])
with tf.GradientTape() as tape:
    g = x[0] ** 2 + x[1] ** 2 - 5 * tf.sin(x[2] ** 2) * tf.tan(x[0] * x[1] * x[2])
    grad = tape.gradient(g, x)
grad.numpy()
