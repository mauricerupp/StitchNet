import l1_loss


from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.utils import plot_model
import tensorflow as tf
import datetime


def create_model(pretrained_weights=None, input_size=None, G0=64, G=32, D=20, C=6):
    """
    RDN implemented from paper 'Residual Dense Network for Image Super-Resolution
    with a custom conv layer which shares weights over all stacked input images
    and a depth-to-space layer at the end of the pipeline
    This is pretty much the network Paolo described in the meeting (the after-resnet part)
    and in the beginning the feature-extractor of Givi
    :param G0: filtersize for the last convolutional layer
    :param G: filtersize per convolutional layer
    :param D: amout of residual dense blocks (RDB)
    :param C: Amount of convolutional layers per RDB
    :param pretrained_weights:
    :param input_size:
    :return:
    """
    inputs = Input(input_size)

    # extract features for every input image
    conv1 = feature_extract(inputs, G0)

    global_conv1 = Conv2D(G0, kernel_size=3, activation='relu', padding='same', name='global_conv1')(conv1)
    global_conv2 = Conv2D(G0, kernel_size=3, activation='relu', padding='same', name='global_conv2')(global_conv1)

    # first RDB
    RDB = create_RDB(global_conv2, 'RDB1', G0, G, C)
    RDBlocks_list = [RDB, ]

    # add the remaining RDB
    for i in range(2, D + 1):
        RDB = create_RDB(RDB, 'RDB' + str(i), G0, G, C)
        RDBlocks_list.append(RDB)

    RDB_out = Concatenate(axis=3)(RDBlocks_list)
    RDB_out = Conv2D(G0, kernel_size=1, padding='same', name='global_1x1_conv')(RDB_out)
    RDB_out = Conv2D(G0, kernel_size=3, padding='same', name='global_conv3')(RDB_out)
    out = Add()([RDB_out, global_conv1])

    # Upscaling / depth to space
    out = Conv2D(G0, kernel_size=3, padding='same', name='upscale_conv_2')(out)
    out = Conv2D(int(G0 / 2), kernel_size=3, padding='same', name='upscale_conv_3')(out)
    out = Conv2D(12, kernel_size=3, padding='same')(out)
    out = depth_to_space(out, 2)

    # fixing layer
    out = Conv2D(3, kernel_size=3, padding='same')(out)
    out = Conv2D(3, kernel_size=3, padding='same')(out)
    out = Conv2D(3, kernel_size=3, padding='same')(out)

    # since we output a color image, we want 3 filters as the last layer
    out = Conv2D(3, kernel_size=3, padding='same')(out)

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


def depth_to_space(input_layer, blocksize):
    """
    implements the tensorflow depth to space function
    :param input_layer:
    :return:
    """
    return Lambda(lambda x: tf.depth_to_space(x, block_size=blocksize, data_format='NHWC'), name='Depth_to_Space',)(input_layer)


def feature_extract(input_tensor, G0):
    """
    Runs every input image seperately through the same Conv-Layer
    and concatenates all tensors at the end
    :param input_tensor: the input layer of the extractor
    :return:
    """
    conv = Conv2D(G0, kernel_size=3, padding='same', activation='relu', name='input_feature_conv')
    input_size = input_tensor.get_shape().as_list()
    in_conv_list = []
    index = 1
    for i in range(0, input_size[3], 3):
        x = Lambda(lambda x: x[:, :, :, i:i+3], name='img_{}'.format(str(index)))(input_tensor)
        in_conv_list.append(conv(x))
        index += 1

    return Concatenate(axis=3, name='conc_img_features')(in_conv_list)
# ------- END -------- #


mod = create_model(input_size=(64,64,15), D=3, C=3)