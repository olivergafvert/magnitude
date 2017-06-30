from magnitude_tf import magnitude, spread, dimension_f
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np

def circle(k):
	''' Returns k number of equidistant points on the unit circle '''
	return make_circles(k)[0]

def line(k):
	''' Returns k number of equidistant points in the interval [0, 1] '''
	return np.arange(0, 1, 1.0/k).reshape((-1, 1))

def plot_curve(y, x=None):
	if not x:
		x = [i for i in range(len(y))]
	plt.plot(x, y)
	plt.show()

if __name__=='__main__':
	U = line(100)
	df = dimension_f(U)
	plot_curve(df)
	

