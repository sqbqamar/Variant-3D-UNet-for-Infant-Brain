import tensorflow as tf


"""This script defines basic operations.
"""



################################################################################
# Basic operations building the network
################################################################################
def Pool3d(inputs, kernel_size, strides):
	"""Performs 3D max pooling."""

	return tf.layers.max_pooling3d(
			inputs=inputs,
			pool_size=kernel_size,
			strides=strides,
			padding='same')


def Deconv3D(inputs, filters, kernel_size, strides, bn=False, act_fn=False, use_bias=False):
	"""Performs 3D deconvolution without bias and activation function."""

	deconv= tf.layers.conv3d_transpose(
			inputs=inputs,
			filters=filters,
			kernel_size=kernel_size,
			strides=strides,
			padding='same',
			use_bias=use_bias,
			kernel_initializer=tf.truncated_normal_initializer())
	if bn:
        	deconv = tf.layers.batch_normalization(
                                inputs=deconv,
                                axis=-1,
                                momentum=0.997,
                                epsilon=1e-5,
                                center=True,
                                scale=True,
                                training=training,
                                fused=True)
	if act_fn:
        	deconv = tf.nn.relu6(deconv)
	return deconv


def Conv3D(inputs, filters, kernel_size, strides, use_bias=False):
	"""Performs 3D convolution without bias and activation function."""

	return tf.layers.conv3d(
			inputs=inputs,
			filters=filters,
			kernel_size=kernel_size,
			strides=strides,
			padding='same',
			use_bias=use_bias,
			kernel_initializer=tf.truncated_normal_initializer())


def Dilated_Conv3D(inputs, filters, kernel_size, dilation_rate, use_bias=False):
	"""Performs 3D dilated convolution without bias and activation function."""

	return tf.layers.conv3d(
			inputs=inputs,
			filters=filters,
			kernel_size=kernel_size,
			strides=1,
			dilation_rate=dilation_rate,
			padding='same',
			use_bias=use_bias,
			kernel_initializer=tf.truncated_normal_initializer())


def BN_ReLU(inputs, training):
	"""Performs a batch normalization followed by a ReLU6."""

	# We set fused=True for a significant performance boost. See
	# https://www.tensorflow.org/performance/performance_guide#common_fused_ops
	inputs = tf.layers.batch_normalization(
				inputs=inputs,
				axis=-1,
				momentum=0.997,
				epsilon=1e-5,
				center=True,
				scale=True,
				training=training, 
				fused=True)

	return tf.nn.relu6(inputs)

def Dense(inputs1, growth_rate=12, training):
    [b, w, h, d, c] = inputs1.get_shape().as_list()
    bn_relu1 = BN_ReLU(inputs1, training)
    conv1 = Conv3D(bn_relu1, growth_rate, kernel_size=3, strides=1, use_bias=True)
    concat1 = tf.concat((inputs1, conv1), axis=4)
    bn_relu2 = BN_ReLU(concat1, training)
    conv2 = Conv3D(bn_relu2, growth_rate, kernel_size=3, strides=1, use_bias=True)
    concat2 = tf.concat((concat1, conv2), axis=4)
    bn_relu3 = BN_ReLU(concat2, training)
    conv3 = Conv3D(bn_relu3, growth_rate, kernel_size=3, strides=1, use_bias=True)
    concat3 = tf.concat((concat2, conv3), axis=4)
    bn_relu4 = BN_ReLU(concat3, training)
    conv4 = Conv3D(bn_relu4, c+3*growth_rate, kernel_size=1, strides=1, use_bias=True)
    return conv4

def res_inc_deconv(inputs, name='res_inc_deconv', training):
    [b, w, h, d, c] = inputs.get_shape().as_list()
    deconv1_1 = Deconv3D(inputs, filters, kernel_size=3, strides=1, use_bias=False)
    
    
    
    
    

