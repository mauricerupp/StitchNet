import l1_loss
from group_normalization import InstanceNormalization


from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.applications import *
from tensorflow.python.keras.utils import plot_model
import tensorflow as tf
import datetime


def create_model(pretrained_weights=None, input_size=None, filter_size=128, block_amount = 12, normalizer=None):
    """
    A simple residual network with ResBlocks from Givi
    :param G0: filtersize for the last convolutional layer
    :param G: filtersize per convolutional layer
    :param D: amout of residual dense blocks (RDB)
    :param C: Amount of convolutional layers per RDB
    :param pretrained_weights:
    :param input_size:
    :return:
    """

    inputs = Input(input_size)

    x = split_input(inputs)
    x = Conv2D(3, kernel_size=5, padding='same', activation='tanh')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=l1_loss.custom_loss, metrics=['accuracy'])
    model.summary()

    # Save the configurations as txt-file
    #with open('RDN ' + str(datetime.datetime.now()) + ' config.txt', 'w') as fh:
    #    model.summary(print_fn=lambda x: fh.write(x + '\n'))

    #plot_model(model, to_file='RN_1.png')

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def split_input(input_tensor):
    """
    Runs every input image seperately through the same Conv-Layer
    and concatenates all tensors at the end
    :param input_tensor: the input layer of the extractor
    :return:
    """
    input_size = input_tensor.get_shape().as_list()
    size_per_img = [input_size[1], input_size[2], 3]
    vgg19layer = ResNet50(include_top=False, input_shape=size_per_img, pooling=None)
    in_conv_list = []
    index = 1
    for i in range(0, input_size[3], 3):
        x = Lambda(lambda x: x[:, :, :, i:i+3], name='img_{}'.format(str(index)))(input_tensor)
        in_conv_list.append(vgg19layer(inputs=x))
        index += 1

    return Concatenate(axis=3, name='conc_img_features')(in_conv_list)


mod = create_model(input_size=(64,64,15), filter_size=128, block_amount=20, normalizer='batch')