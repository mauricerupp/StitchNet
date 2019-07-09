from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
from tensorflow.python.keras.layers import *
import random
import cv2
import tensorflow.keras.backend as K
from tensorflow.python.keras import initializers, regularizers, constraints
import numpy as np


# ---- Functions ---- #
def preprocess_to_caffe(x):

    # 'RGB'->'BGR'
    x = x[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    std = None

    _IMAGENET_MEAN = K.constant(-np.array(mean))

    # Zero-center by mean pixel
    if K.dtype(x) != K.dtype(_IMAGENET_MEAN):
        x = K.bias_add(
            x, K.cast(_IMAGENET_MEAN, K.dtype(x)),
            data_format='channels_last')
    else:
        x = K.bias_add(x, _IMAGENET_MEAN, 'channels_last')
    if std is not None:
        x /= std
    return x


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def zero_center(in_img):
    """
    :param in_img: an image in the scale of [0,1]
    :return: an image in the scale of [-1,1]
    """
    return 2 * in_img - 1


def revert_zero_center(in_img):
    return in_img / 2 + 0.5


def random_numpy_crop(in_img, crop_size):
    img_size = in_img.shape
    assert img_size[2] == crop_size[2]

    # if the image is too small rescale it to atleast the crop_size
    if img_size[0] <= crop_size[0] or img_size[1] <= crop_size[1]:
        in_img = scale_img(in_img, crop_size)
        img_size = in_img.shape

    # if the image is too big, we scale it down in order to have some detail in the image and not only blurry background
    if img_size[0] > 5*crop_size[0] or img_size[1] > 5*crop_size[1]:
        new_size = np.array(crop_size)

        new_size[0] = 2*new_size[0]
        new_size[1] = 2*new_size[1]
        in_img = scale_img(in_img, new_size)
        img_size = in_img.shape

    top_left_corner = [random.randint(0, int(img_size[0]-crop_size[0])),
                       random.randint(0, int(img_size[1]-crop_size[1]))]

    out = in_img[top_left_corner[0]:top_left_corner[0]+crop_size[0],
           top_left_corner[1]:top_left_corner[1]+crop_size[1], :]
    assert out.shape[0] == crop_size[0]
    assert out.shape[1] == crop_size[1]
    assert out.shape[2] == crop_size[2]
    return out


def scale_img(img, des_size):
    old_size = img.shape[:2]
    h_ratio = des_size[0] / old_size[0]
    w_ratio = des_size[1] / old_size[1]

    # find out how to scale it, so that it still covers the whole area
    if h_ratio * old_size[1] > des_size[1]:
        ratio = h_ratio
    else:
        ratio = w_ratio

    # resize the image while keeping its ratio so its not warped
    new_size = tuple([int(round(x*ratio)) for x in old_size])
    return cv2.resize(img, (new_size[1], new_size[0]))


def resize_img(img, desired_size):
    # scale the image until it covers the whole desired size
    img = scale_img(img, desired_size)

    (h, w) = img.shape[:2]
    # cut the image so that the cut frame is centered
    h_cut = h - desired_size[0]
    space_h = int(h_cut / 2)
    w_cut = w - desired_size[1]
    space_w = int(w_cut / 2)

    assert h_cut >= 0
    assert w_cut >= 0

    img = img[space_h:space_h+desired_size[0], space_w:space_w+desired_size[1]]

    return img


def crop(img, center, size):
    """
    crops an image around a given center to a given size
    if the size around this center is not inside the image borders it shifts to the border
    :param img: the img we want to crop
    :param center: the new center
    :param size: the new size
    :return: a cropped image
    """
    assert center[0] >= 0
    assert center[1] >= 0
    (h, w) = img.shape[:2]
    (up, under, left, right) = (int(center[0]-size[0] // 2), int(center[0] + size[0] - (size[0] // 2)),
                                int(center[1] - size[1] // 2), int(center[1] + size[1] - (size[1] // 2)))
    if up >= 0:
        if under <= h:
            pass
        else:
            under = h
            up = under - size[0]
    else:
        up = 0
        under = size[0]

    if left >= 0:
        if right <= w:
            pass
        else:
            right = w
            left = right - size[1]
    else:
        left = 0
        right = size[1]
    img = img[up:under, left:right]
    assert img.shape[:2] == (size[0], size[1])
    return img


# ---- Layers ---- #
def depth_to_space(input_layer, blocksize):
    """
    implements the tensorflow depth to space function
    """
    return Lambda(lambda x: tf.depth_to_space(x, block_size=blocksize, data_format='NHWC'), name='Depth_to_Space',)(input_layer)


def create_resblock(prior_layer, block_name, n_filters, kernel_size, stride, dilation, normalizer):
    x = Conv2D(filters=n_filters, kernel_size=kernel_size,
               strides=stride, dilation_rate=dilation,
               name=block_name + "_conv1",
               padding='same')(prior_layer)
    x = normalize(x, name=block_name + "_norm1", normalizer=normalizer)
    x = Activation('relu')(x)

    x = Conv2D(filters=n_filters, kernel_size=kernel_size,
               strides=stride, dilation_rate=dilation,
               name=block_name + "_conv2",
               padding='same')(x)

    x = normalize(x, name=block_name + "_norm2", normalizer=normalizer)

    return Add()([prior_layer, x])


def create_RDB(prior_layer, block_name, G0=64, G=32, C=6):
    """
    Creates one residual dense block
    :param prior_layer: the layer that was there before the block
    :param block_name: name of the RDB
    :param G0: filtersize for the last convolutional layer
    :param G: filtersize per convolutional layer
    :param C: Amount of convolutional layers per RDB
    :return: a residual dense block with C conv layers
    """
    layerlist = [prior_layer]
    conv_layer = Conv2D(G, kernel_size=3, padding='same', activation='relu', name=block_name+'_conv1')(prior_layer)

    # iterating from 2 on in order to have a clear labelling of the layers
    for i in range(2, C + 1):
        layerlist.append(conv_layer)
        out = Concatenate(axis=3)(layerlist)  # concatenate the output over the color channel
        conv_layer = Conv2D(G, kernel_size=3, padding='same', activation='relu', name=block_name+'_conv'+str(i))(out)

    # append the last convolutional layer of the RDB to the rest of conv layers
    layerlist.append(conv_layer)
    out = Concatenate(axis=3, name=block_name+'_conc')(layerlist)
    # the last conv layer which has a kernel_size of 1
    feat = Conv2D(G0, kernel_size=1, padding='same', activation='relu', name=block_name+'_local_Conv')(out)
    feat = Add()([feat, prior_layer])
    return feat


def feature_extract(input_tensor, filter, kernel):
    """
    Runs every input image seperately through the same Conv-Layer
    and concatenates all tensors at the end
    :param G0: filter size of the Convolutional layer
    :param input_tensor: the input layer of the extractor
    :return:
    """
    conv = Conv2D(filter, kernel_size=kernel, padding='same', activation='relu', name='input_feature_conv')
    input_size = input_tensor.get_shape().as_list()
    in_conv_list = []
    index = 1
    for i in range(0, input_size[3], 3):
        x = Lambda(lambda x: x[:, :, :, i:i+3], name='img_{}'.format(str(index)))(input_tensor)
        in_conv_list.append(conv(x))
        index += 1

    return Concatenate(axis=3, name='conc_img_features')(in_conv_list)


def encode(input_tensor, kernel):
    """
    a simple encoder which uses strided convolutions and an increasing amount of filters
    conv + norm (add trainable variables to graph colleciton) + activation
    :param input_tensor:
    """

    conv = Conv2D(64, kernel, activation='relu', padding='same', name='encoder_conv1')(input_tensor)
    conv = Conv2D(64, kernel, activation='relu', padding='same', name='encoder_conv2')(conv)
    conv = Conv2D(128, kernel, activation='relu', padding='same', name='encoder_conv3', strides=2)(conv)
    conv = Conv2D(128, kernel, activation='relu', padding='same', name='encoder_conv4')(conv)
    conv = Conv2D(256, kernel, activation='relu', padding='same', name='encoder_conv5', strides=2)(conv)
    conv = Conv2D(256, kernel, activation='relu', padding='same', name='encoder_conv6')(conv)
    conv = Conv2D(256, kernel, activation='relu', padding='same', name='encoder_conv7', strides=2)(conv)
    conv = Conv2D(256, kernel, activation='relu', padding='same', name='encoder_conv8')(conv)
    conv = Conv2D(256, kernel, activation='relu', padding='same', name='encoder_conv9', strides=2)(conv)
    return Conv2D(256, kernel, activation='relu', padding='same', name='encoder_conv10')(conv)


def single_decode(input_tensor, kernel):
    """
    a decoder fitting for function encode
    :param input_tensor:
    """
    conv = Conv2DTranspose(128, kernel, activation='relu', padding='same', name='decoder_conv1', strides=2)(input_tensor)
    conv = Conv2D(128, kernel, activation='relu', padding='same', name='decoder_conv2')(conv)
    conv = Conv2DTranspose(64, kernel, activation='relu', padding='same', name='decoder_conv3', strides=2)(conv)
    conv = Conv2D(64, kernel, activation='relu', padding='same', name='decoder_conv4')(conv)
    conv = Conv2DTranspose(32, kernel, activation='relu', padding='same', name='decoder_conv5', strides=2)(conv)
    conv = Conv2D(32, kernel, activation='relu', padding='same', name='decoder_conv6')(conv)
    conv = Conv2DTranspose(16, kernel, activation='relu', padding='same', name='decoder_conv7', strides=2)(conv)
    conv = Conv2D(16, kernel, activation='relu', padding='same', name='decoder_conv8')(conv)
    return Conv2DTranspose(3, kernel, activation='tanh', padding='same', name='decoder_conv9', strides=1)(conv)


def enc_block(input_layer, filters, index, normalizer, isTraining):
    if filters > 256:
        filters = 256
    if index == 1:
        x = Conv2D(filters, 3, activation=None, padding='same', strides=1, name='encoder_conv{}_1'.format(index))(input_layer)
        x = Activation('relu')(x)
    else:
        x = Conv2D(filters, 3, activation=None, padding='same', strides=1, name='encoder_conv{}_1'.format(index))(input_layer)
        x = normalize(x, 'encoder_norm{}_1'.format(index), normalizer, isTraining)
        x = Activation('relu')(x)

    x = Conv2D(filters, 3, activation=None, padding='same', strides=1, name='encoder_conv{}_2'.format(index))(x)
    x = normalize(x, 'encoder_norm{}_2'.format(index), normalizer, isTraining)
    x = Activation('relu')(x)
    if filters < 512:
        x = Conv2D(filters*2, 3, activation=None, padding='same', strides=2, name='encoder_conv{}_3'.format(index))(x)
    else:
        x = Conv2D(filters, 3, activation=None, padding='same', strides=2, name='encoder_conv{}_3'.format(index))(x)
    x = normalize(x, 'encoder_norm{}_3'.format(index), normalizer, isTraining)
    return Activation('relu')(x)


def dec_block(input_layer, filters, index, normalizer, isTraining, modelname):
    x = Conv2DTranspose(filters, 3, activation=None, padding='same', name='{}_decoder_conv{}_1'.format(modelname, index), strides=2)(input_layer)
    x = normalize(x, '{}_decoder_norm{}_1'.format(modelname, index), normalizer, isTraining)
    x = Activation('relu')(x)
    for i in range(2):
        x = Conv2D(filters, 3, activation=None, padding='same', strides=1, name='{}_decoder_conv{}_{}'.format(modelname, index, i +2))(x)
        x = normalize(x, '{}_decoder_norm{}_{}'.format(modelname, index, i+2), normalizer, isTraining)
        x = Activation('relu')(x)

    return x


# ---- Normalization ---- #
def normalize(input_layer, name, normalizer, training_flag):
    if normalizer.lower() == 'batch':
        return BatchNormalization(name=name, axis=3, trainable=training_flag)(input_layer)
    elif normalizer.lower() == 'instance':
        return InstanceNormalization(axis=3)(input_layer)
    else:
        print("No valid normalize")
        exit()


class InstanceNormalization(tf.keras.layers.Layer):
    """Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
