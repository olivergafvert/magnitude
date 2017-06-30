#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 16:59:35 2017

@author: oliver
"""
import tensorflow as tf
import numpy as np

INFINITY = float("inf")


def euclidean_distance_matrix(U):
	U = tf.reshape(U, [tf.shape(U)[0], 1, -1])
	U = U*tf.ones([1, tf.shape(U)[0], tf.shape(U)[2]])
	D = U-tf.transpose(U, perm=[1, 0, 2])
	return tf.sqrt(tf.reduce_sum(tf.multiply(D, D), axis=2))

def manhattan_distance_matrix(U):
	U = tf.reshape(U, [tf.shape(U)[0], 1, -1])
	U = U*tf.ones([1, tf.shape(U)[0], tf.shape(U)[2]])
	D = U-tf.transpose(U, perm=[1, 0, 2])
	return tf.reduce_sum(tf.abs(D), axis=2)

def knn_matrix(D):
    return tf.map_fn(lambda v: tf.map_fn(lambda e: count_lt(v, e)-1, v), D)

def count_lt(v, e):
	return tf.reduce_sum(tf.cast(tf.less_equal(v, e), tf.float32))

def _diff(m):
	''' Computes an approximation of the derivative dlog(|tX|)/dlog(t) '''
	m = np.log(m)
	m = m-np.roll(m, 1)
	return np.divide(m[1, 1:], m[0, 1:])

def _magnitude(D, t):
	''' Computes the magnitude |tX|, as described in [1], given the distance matrix D and parameter t

		[1] -  '''
	b = tf.ones([tf.shape(D)[0]])
	return tf.reduce_sum(tf.matrix_inverse(tf.exp(-t*D)))

def _spread(D, t, q):
	''' Computes the q-spread E_q(X, t) as described in [1] given the distance matrix D and parameters t and q 

		[1] - "Spread: a measure of the size of metric spaces", Simon Willerton'''
	N = tf.cast(tf.shape(D)[0], tf.float32)
	if q==1:
		return tf.reduce_prod(tf.pow(tf.reduce_sum(tf.exp(-t*D), axis=1), 1/N))*N
	elif q==INFINITY:
		return tf.minimum(tf.reciprocal(tf.reduce_sum(tf.exp(-t*D), axis=1)))*N
	return tf.pow(tf.reduce_sum(tf.pow(tf.reduce_sum(tf.exp(-t*D), axis=1), q-1)), 1.0/(1-q))\
		*tf.pow(tf.cast(tf.shape(D)[0], tf.float32), q/(q-1))

def init_graph(distance='euclidean'):
	''' Constructs the computation graph '''
	U = tf.placeholder(tf.float32, [None, None])
	if distance=='euclidean':
		D = euclidean_distance_matrix(U)
	elif distance=='manhattan':
		D = manhattan_distance_matrix(U)
	else:
		Exception("Failed to parse distance function. Exiting...")
		return
	t = tf.placeholder(tf.float32)
	m = _magnitude(D, t)
	q = tf.placeholder(tf.float32)
	s = _spread(D, t, q)
	init = tf.global_variables_initializer()
	return init, U, t, m, s, q

def magnitude(_U, distance='euclidean', indices=np.arange(0.01, 5, 0.01)):
	''' Computes the magnitude |tX| for all t in the set indices (see @_magnitude).

		_U - a matrix where each row is a point in R^n
		distance - a string representing the metric that should be used ['euclidean', 'manhattan']
		indices - an array of indices for which the magnitude should be computed '''
	init, U, t, m, _, _ = init_graph(distance)
	if not init:
		return
	with tf.Session() as sess:
		sess.run(init)
		values = np.zeros(len(indices))
		for i in range(len(indices)):
			values[i] = sess.run(m, feed_dict={U: _U, t: indices[i]})
		return values

def spread(_U, distance='euclidean', indices=np.arange(0.01, 5, 0.01), _q=0):
	''' Computes the q-spread E_q(X, t) for all t in the set indices (see @_spread).

		_U - a matrix where each row is a point in R^n
		distance - a string representing the metric that should be used ['euclidean', 'manhattan']
		indices - an array of indices for which the spread should be computed
		_q - which type of spread that should be computed (default=0) '''
	init, U, t, _, s, q = init_graph(distance)
	if not init:
		return
	with tf.Session() as sess:
		sess.run(init)
		values = np.zeros(len(indices))
		for i in range(len(indices)):
			values[i] = sess.run(s, feed_dict={U: _U, t: indices[i], q: _q})
		return values

def dimension_f(U, distance='euclidean', indices=np.exp(np.arange(0.01, 5, 0.01)), q=0):
	return _diff(np.vstack((indices, spread(U, distance, indices, q))))



''' Multidimensional extension '''
def _m_diff(M, indices):
	''' Computes the multidimensional dimension function '''
	h_diff = np.array(map(lambda x: diff(np.vstack((indices, x))), M[:-1]))
	v_diff = np.array([diff(np.vstack((indices, M[:, i]))) for i in range(np.size(M, 1)-1)]).T
	return np.sqrt(np.power(h_diff, 2)+np.power(v_diff, 2))

def _m_magnitude(D, Dp, t, s):
	b = tf.ones([tf.shape(D)[0]])
	return tf.reduce_sum(tf.matrix_inverse(tf.exp(-t*D-s*Dp)))

def _m_spread(D, Dp, t, s, q):
	N = tf.cast(tf.shape(D)[0], tf.float32)
	if q==1:
		return tf.reduce_prod(tf.pow(tf.reduce_sum(tf.exp(-t*D-s*Dp), axis=1), 1/N))*N
	elif q==INFINITY:
		return tf.minimum(tf.reciprocal(tf.reduce_sum(tf.exp(-t*D-s*Dp), axis=1)))*N
	return tf.pow(tf.reduce_sum(tf.pow(tf.reduce_sum(tf.exp(-t*D-s*Dp), axis=1), q-1)), 1.0/(1-q))\
		*tf.pow(tf.cast(tf.shape(D)[0], tf.float32), q/(q-1))
