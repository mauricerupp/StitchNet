import l1_loss


from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.utils import plot_model
import tensorflow as tf
import datetime

# TODO: implement instancenorm, global skips, general the numbers

def create_model(pretrained_weights=None, input_size=None, filter_size=128, block_amount = 12):
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

    global_conv1 = Conv2D(filter_size, kernel_size=3, activation='relu', padding='same', name='global_conv1')(inputs)
    global_conv2 = Conv2D(filter_size, kernel_size=3, activation='relu', padding='same', name='global_conv2')(global_conv1)

    # first RB
    RB = create_resblock(prior_layer=global_conv2, block_name='RB1',
                         n_filters=filter_size, kernel_size=3, stride=1, dilation=1, normalizer=BatchNormalization)

    # add the remaining RDB
    for i in range(2, block_amount + 1):
        RB = create_resblock(prior_layer=RB, block_name='RB' + str(i),
                             n_filters=filter_size, kernel_size=3, stride=1, dilation=1, normalizer=BatchNormalization)


    # Upscaling / depth to space
    out = depth_to_space(RB, 2)

    # since we output a color image, we want 3 filters as the last layer
    out = Conv2D(3, kernel_size=3, padding='same', activation='tanh')(out)

    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer='adam', loss=l1_loss.custom_loss, metrics=['accuracy'])
    model.summary()

    # Save the configurations as txt-file
    #with open('RDN ' + str(datetime.datetime.now()) + ' config.txt', 'w') as fh:
    #    model.summary(print_fn=lambda x: fh.write(x + '\n'))

    #plot_model(model, to_file='RDN_1_D{}C{}.png'.format(D, C))

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


# ------- Functions -------- #
def create_resblock(prior_layer, block_name, n_filters, kernel_size, stride, dilation, normalizer):
    x = Conv2D(filters=n_filters, kernel_size=kernel_size,
               strides=stride, dilation_rate=dilation,
               name=block_name + "_conv1",
               padding='same')(prior_layer)
    x = normalizer(name=block_name + "_norm1")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=n_filters, kernel_size=kernel_size,
               strides=stride, dilation_rate=dilation,
               name=block_name + "_conv2",
               padding='same')(x)

    x = normalizer(name=block_name + "_norm2")(x)

    return Add()([prior_layer, x])


def depth_to_space(input_layer, blocksize):
    """
    implements the tensorflow depth to space function
    :param input_layer:
    :return:
    """
    return Lambda(lambda x: tf.depth_to_space(x, block_size=blocksize, data_format='NHWC'), name='Depth_to_Space',)(input_layer)
# ------- END -------- #


mod = create_model(input_size=(64,64,15), filter_size=128, block_amount=12)