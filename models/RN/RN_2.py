import l1_loss
from utilities import *
from RDN_1 import feature_extract


from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.utils import plot_model
import tensorflow as tf
import datetime


def create_model(pretrained_weights=None, input_size=None, filter_size=128, block_amount = 12, normalizer=None, isTraining=True):
    """
    A simple residual network with ResBlocks from Givi and added the feature extraction layer of RDNs in the beginning
    :param G0: filtersize for the last convolutional layer
    :param G: filtersize per convolutional layer
    :param D: amout of residual dense blocks (RDB)
    :param C: Amount of convolutional layers per RDB
    :param pretrained_weights:
    :param input_size:
    :return:
    """
    inputs = Input(input_size)

    # feature extractor of RDNs
    conv1 = feature_extract(inputs, 64, 3)
    # first global conv has no normalization
    global_conv1 = Conv2D(filter_size*2, kernel_size=7, activation='relu', padding='same', name='global_conv1')(conv1)

    global_conv2 = Conv2D(filter_size, kernel_size=7, padding='same', name='global_conv2')(global_conv1)
    global_conv2 = normalize(name="globalconv2_norm1", input_layer=global_conv2, normalizer=normalizer, training_flag=isTraining)
    global_conv2 = Activation('relu')(global_conv2)

    # first RB
    RB = create_resblock(prior_layer=global_conv2, block_name='RB1',
                         n_filters=filter_size, kernel_size=3, stride=1, dilation=1, normalizer=normalizer, isTraining=isTraining)

    # add the remaining RDB
    for i in range(2, block_amount + 1):
        RB = create_resblock(prior_layer=RB, block_name='RB' + str(i),
                             n_filters=filter_size, kernel_size=3, stride=1, dilation=1, normalizer=normalizer, isTraining=isTraining)

    # depth to space
    out = depth_to_space(RB, 2)

    # TODO: Add layers here?
    out = Conv2D(16, kernel_size=5, padding='same', activation='relu')(out)
    out = Conv2D(8, kernel_size=5, padding='same', activation='relu')(out)

    # since we output a color image, we want 3 filters as the last layer
    out = Conv2D(3, kernel_size=5, padding='same', activation='tanh')(out)

    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer='adam', loss=l1_loss.custom_loss, metrics=['accuracy'])
    model.summary()

    # Save the configurations as txt-file
    #with open('RDN ' + str(datetime.datetime.now()) + ' config.txt', 'w') as fh:
    #    model.summary(print_fn=lambda x: fh.write(x + '\n'))

    #plot_model(model, to_file='RN_1.png')

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


mod = create_model(input_size=(64,64,15), filter_size=128, block_amount=20, normalizer='instance')