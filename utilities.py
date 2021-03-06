from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.keras.layers import *
import random
import cv2
import tensorflow.keras.backend as K
from tensorflow.python.keras import initializers, regularizers, constraints
import numpy as np

"""
This is a file where all the shared utility functions and layers are stored
"""


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

    top_left_corner = [random.randint(0, int(img_size[0]-crop_size[0])),
                       random.randint(0, int(img_size[1]-crop_size[1]))]

    out = in_img[top_left_corner[0]:top_left_corner[0]+crop_size[0],
           top_left_corner[1]:top_left_corner[1]+crop_size[1], :]

    # flip the image from right to left w/ 50% chance
    flipit = random.choice([True, False])
    if flipit:
        out = np.fliplr(out)

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
    """
    Difference to scale_img(): Where scale_img just upscales an image with a certain factor
    resize_img() resize an image to a fixed size and eventually cuts away a part of the image in order not to wrap it
    :param img:
    :param desired_size:
    :return:
    """
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


def image_predictor(epoch, logs):
    """
    creates a tester, which predicts a few images after a certain amount of epochs and stores them as png
    we take 4 from the training and 4 from the validation set
    :param epoch:
    :param logs: has to be given as argument in order to compile
    """
    if epoch % 10 == 0:  # print samples every 10 images
        for i in range(0,25):
            # load X
            set = ""
            if i % 2 == 0:
                list = np.load('/data/cvg/maurice/unprocessed/train_snaps_paths.npy')
                set += "train-"
            else:
                list = np.load('/data/cvg/maurice/unprocessed/val_snaps_paths.npy')
                set += "test-"

            # create the path
            if DATASET == "S1":
                loaded_data = create_fixed_path(list[i])
            elif DATASET == "S2":
                loaded_data = create_smooth_rand_path(list[i])
            else:
                loaded_data = create_very_rand_path(list[i])

            # preprocess x
            x = loaded_data[0]
            x = np.expand_dims(x, axis=0)

            # preprocess y
            y_true = loaded_data[1]
            covered_area = y_true[:, :, -3:]
            y_true = y_true[:, :, :-3]
            y_true = revert_zero_center(y_true) * 255
            y_true = np.array(np.rint(y_true), dtype=int)
            covered_target = y_true * covered_area
            covered_target = np.array(np.rint(covered_target), dtype=int)

            # predict y (since the model is trained on pictures in [-1,1], the post-processing reverts it to [0,255])
            y_pred = model.predict(x)
            y_pred = revert_zero_center(y_pred)*255
            y_pred = np.array(np.rint(y_pred), dtype=int)

            # save the result
            fig = plt.figure()
            fig.suptitle('Results of predicting {}Image {} \non epoch {}'.format(set, i, epoch + 1), fontsize=20)
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.set_title('Y_True')
            plt.imshow(y_true[..., ::-1], interpolation='nearest') # conversion to RGB
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.set_title('Y_True covered')
            plt.imshow(covered_target[..., ::-1], interpolation='nearest')
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.set_title('Prediction of model')
            plt.imshow(y_pred[0][..., ::-1], interpolation='nearest')
            plt.savefig("/data/cvg/maurice/logs/{}/Prediction-img{}-epoch{}.png".format(NAME, i, epoch + 1))
            plt.close()



# ---- Layers ---- #
def depth_to_space(input_layer, blocksize):
    """
    implements the tensorflow depth to space function
    """
    return Lambda(lambda x: tf.depth_to_space(x, block_size=blocksize, data_format='NHWC'), name='Depth_to_Space',)(input_layer)


def create_resblock(prior_layer, block_name, n_filters, kernel_size, stride, dilation, normalizer, isTraining=True):
    """
    creates a resblock with two convolution layers and one skip connection
    """
    x = Conv2D(filters=n_filters, kernel_size=kernel_size,
               strides=stride, dilation_rate=dilation,
               name=block_name + "_conv1",
               padding='same')(prior_layer)
    x = normalize(x, name=block_name + "_norm1", normalizer=normalizer, training_flag=isTraining)
    x = Activation('relu')(x)

    x = Conv2D(filters=n_filters, kernel_size=kernel_size,
               strides=stride, dilation_rate=dilation,
               name=block_name + "_conv2",
               padding='same')(x)

    x = normalize(x, name=block_name + "_norm2", normalizer=normalizer, training_flag=isTraining)

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

    # the last conv layer which has a kernel_size of 1 as in the paper
    feat = Conv2D(G0, kernel_size=1, padding='same', activation='relu', name=block_name+'_local_Conv')(out)

    # the last skip connection
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


def enc_block_leaky(input_layer, filters, index, normalizer, isTraining):
    """
    An encoderblock with three convolutions used inside the autoencoder with leakyRelu activation functions
    and a filtersize, which is restrained to 512.
    """
    if filters > 256:
        filters = 256
    if index == 1:
        x = Conv2D(filters, 3, activation=None, padding='same', strides=1, name='encoder_conv{}_1'.format(index))(input_layer)
        x = LeakyReLU()(x)
    else:
        x = Conv2D(filters, 3, activation=None, padding='same', strides=1, name='encoder_conv{}_1'.format(index))(input_layer)
        x = normalize(x, 'encoder_norm{}_1'.format(index), normalizer, isTraining)
        x = LeakyReLU()(x)

    x = Conv2D(filters, 3, activation=None, padding='same', strides=1, name='encoder_conv{}_2'.format(index))(x)
    x = normalize(x, 'encoder_norm{}_2'.format(index), normalizer, isTraining)
    x = LeakyReLU()(x)
    if filters < 512:
        x = Conv2D(filters*2, 3, activation=None, padding='same', strides=2, name='encoder_conv{}_3'.format(index))(x)
    else:
        x = Conv2D(filters, 3, activation=None, padding='same', strides=2, name='encoder_conv{}_3'.format(index))(x)
    x = normalize(x, 'encoder_norm{}_3'.format(index), normalizer, isTraining)

    return LeakyReLU()(x)


def dec_block_leaky(input_layer, filters, index, normalizer, isTraining, modelname):
    """
    A decoderblock with three convolutions used inside the autoencoder with leakyRelu activation functions.
    """
    x = Conv2DTranspose(filters, 3, activation=None, padding='same', name='{}_decoder_conv{}_1'.format(modelname, index), strides=2)(input_layer)
    x = normalize(x, '{}_decoder_norm{}_1'.format(modelname, index), normalizer, isTraining)
    x = LeakyReLU()(x)
    for i in range(2):
        x = Conv2D(filters, 3, activation=None, padding='same', strides=1, name='{}_decoder_conv{}_{}'.format(modelname, index, i +2))(x)
        x = normalize(x, '{}_decoder_norm{}_{}'.format(modelname, index, i+2), normalizer, isTraining)
        x = LeakyReLU()(x)

    return x


def big_dec_block_leaky(input_layer, filters, index, normalizer, isTraining, modelname):
    x = Conv2DTranspose(filters, 3, activation=None, padding='same', name='{}_decoder_conv{}_1'.format(modelname, index), strides=2)(input_layer)
    x = normalize(x, '{}_decoder_norm{}_1'.format(modelname, index), normalizer, isTraining)
    x = LeakyReLU()(x)
    for i in range(3):
        x = Conv2D(filters, 3, activation=None, padding='same', strides=1, name='{}_decoder_conv{}_{}'.format(modelname, index, i +2))(x)
        x = normalize(x, '{}_decoder_norm{}_{}'.format(modelname, index, i+2), normalizer, isTraining)
        x = LeakyReLU()(x)

    return x


# ---- Normalization Layers ---- #
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
