# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 10:52:33 2018

@author: SAQIB QAMAR
"""
import tensorflow as tf
from utils import Deconv3D, Conv3D, BN_ReLU, Dilated_Conv3D
import numpy as np

c = [32,68,104,140]
def Dense(inputs, growth_rate, training):
#	[b, w, h, d, c] = inputs.get_shape().as_list()
#	c = [32,68,104,140]
	bn_relu1 = BN_ReLU(inputs, training)
	conv1 = Conv3D(bn_relu1, growth_rate, 3, 1)
	concat1 = tf.concat((inputs, conv1), axis=4)
	bn_relu2 = BN_ReLU(concat1, training)
	conv2 = Conv3D(bn_relu2, growth_rate, kernel_size=3, strides=1)
	concat2 = tf.concat((concat1, conv2), axis=4)
	bn_relu3 = BN_ReLU(concat2, training)
	conv3 = Conv3D(bn_relu3, growth_rate, kernel_size=3, strides=1)
	concat3 = tf.concat((concat2, conv3), axis=4)
	bn_relu4 = BN_ReLU(concat3, training)
	conv4 = Conv3D(bn_relu4, c[0]+3*growth_rate, kernel_size=1, strides=1)
	return conv4

def Dense1(inputs, growth_rate, training):
#	[b, w, h, d, c] = inputs.get_shape().as_list()
	bn_relu1 = BN_ReLU(inputs, training)
	conv1 = Conv3D(bn_relu1, growth_rate, 3, 1)
	concat1 = tf.concat((inputs, conv1), axis=4)
	bn_relu2 = BN_ReLU(concat1, training)
	conv2 = Conv3D(bn_relu2, growth_rate, kernel_size=3, strides=1)
	concat2 = tf.concat((concat1, conv2), axis=4)
	bn_relu3 = BN_ReLU(concat2, training)
	conv3 = Conv3D(bn_relu3, growth_rate, kernel_size=3, strides=1)
	concat3 = tf.concat((concat2, conv3), axis=4)
	bn_relu4 = BN_ReLU(concat3, training)
	conv4 = Conv3D(bn_relu4, c[1]+3*growth_rate, kernel_size=1, strides=1)
	return conv4

def Dense2(inputs, growth_rate, training):
#	[b, w, h, d, c] = inputs.get_shape().as_list()
	bn_relu1 = BN_ReLU(inputs, training)
	conv1 = Conv3D(bn_relu1, growth_rate, 3, 1)
	concat1 = tf.concat((inputs, conv1), axis=4)
	bn_relu2 = BN_ReLU(concat1, training)
	conv2 = Conv3D(bn_relu2, growth_rate, kernel_size=3, strides=1)
	concat2 = tf.concat((concat1, conv2), axis=4)
	bn_relu3 = BN_ReLU(concat2, training)
	conv3 = Conv3D(bn_relu3, growth_rate, kernel_size=3, strides=1)
	concat3 = tf.concat((concat2, conv3), axis=4)
	bn_relu4 = BN_ReLU(concat3, training)
	conv4 = Conv3D(bn_relu4, c[2]+3*growth_rate, kernel_size=1, strides=1)
	return conv4

def Dense3(inputs, growth_rate, training):
#	[b, w, h, d, c] = inputs.get_shape().as_list()
	bn_relu1 = BN_ReLU(inputs, training)
	conv1 = Conv3D(bn_relu1, growth_rate, 3, 1)
	concat1 = tf.concat((inputs, conv1), axis=4)
	bn_relu2 = BN_ReLU(concat1, training)
	conv2 = Conv3D(bn_relu2, growth_rate, kernel_size=3, strides=1)
	concat2 = tf.concat((concat1, conv2), axis=4)
	bn_relu3 = BN_ReLU(concat2, training)
	conv3 = Conv3D(bn_relu3, growth_rate, kernel_size=3, strides=1)
	concat3 = tf.concat((concat2, conv3), axis=4)
	bn_relu4 = BN_ReLU(concat3, training)
	conv4 = Conv3D(bn_relu4, c[3]+3*growth_rate, kernel_size=1, strides=1)
	return conv4

def res_inc_deconv(inputs, training):
#	[b, w, h, d, c]= inputs.get_shape().as_list()
	deconv1_1_1 = Deconv3D(inputs, 32, kernel_size=3, strides=1, use_bias=False)#44
	deconv1_1 = BN_ReLU(deconv1_1_1, training)
	deconv2_1_1 = Deconv3D(inputs, 32, kernel_size=3, strides=1, use_bias=False)#44
	deconv2_1 = BN_ReLU(deconv2_1_1, training)
	deconv2_3 = Deconv3D(deconv2_1, 64, kernel_size=3, strides=1, use_bias=False)#88
	deconv2_2 = BN_ReLU(deconv2_3, training)
	deconv3_1_1 = Deconv3D(inputs, 32, kernel_size=3, strides=1, use_bias=False)#44
	deconv3_1 = BN_ReLU(deconv3_1_1, training)
	deconv3_2_1 = Dilated_Conv3D(deconv3_1, 32, kernel_size=3, dilation_rate=2, use_bias=False)#44
	deconv3_2 = BN_ReLU(deconv3_2_1, training)
	concat = tf.concat((deconv1_1, deconv2_2, deconv3_2), axis=4)
	deconv1 = Deconv3D(concat, 128, kernel_size=3, strides=1, use_bias=False)#176
	deconv= BN_ReLU(deconv1, training)
	fuse = tf.add(inputs, deconv)
	return fuse

def unpool(inputs, training):
#	[b, w, h, d, c] = inputs.get_shape().as_list()
	conv31 = Conv3D(inputs, 176, kernel_size=3, strides=1)
	deconv31= BN_ReLU(conv31, training)
	deconv1_1 = Deconv3D(deconv31, 176, kernel_size=3, strides=1, use_bias=False)
	deconv1= BN_ReLU(deconv1_1, training)
	deconv1_2 = Deconv3D(deconv1, 88, kernel_size=3, strides=2, use_bias=False)
	deconv2= BN_ReLU(deconv1_2, training)
	deconv2_1 = Deconv3D(inputs, 176, kernel_size=3, strides=1, use_bias=False)
	deconv3= BN_ReLU(deconv2_1, training)
	deconv2_2 = Dilated_Conv3D(deconv3, 176, kernel_size=3, dilation_rate=2, use_bias=False)
	deconv4= BN_ReLU(deconv2_2, training)
	deconv2_3 = Deconv3D(deconv4, 88, kernel_size=3, strides=2, use_bias=False)
	deconv5= BN_ReLU(deconv2_3, training)
	concat = tf.concat((deconv2, deconv5), axis=4)
	return concat
