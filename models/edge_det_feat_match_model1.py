import l1_loss

from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.utils import plot_model


#creating a residual block
def residual_block(x_input):

    conv1 = Conv2D(64, kernel_size=(3,3), strides=(1, 1), padding='same')(x_input)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)

    conv2 = Conv2D(64, kernel_size=(3,3), strides=(1, 1), padding='same')(act1)
    bn2 = BatchNormalization()(conv2)

    return Add()([bn2, Activation('relu')(x_input), x_input])


def create_model(pretrained_weights=None, input_size=None):
    """
    uses the ECNN from paper 'A Generic deep architecture for single image reflection removal and...'

    :param pretrained_weights:
    :param input_size:
    :return:
    """
    # ECNN-Part:
    x_input = Input(input_size)

    conv1 = Conv2D(64, kernel_size=(3,3), strides=(1, 1), padding='same')(x_input)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)

    conv2 = Conv2D(64, kernel_size=(3,3), strides=(1, 1), padding='same')(act1)
    bn2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(bn2)

    pad3 = ZeroPadding2D(padding=(1,1))(act2)
    conv3 = Conv2D(64, kernel_size=(3,3), strides=(2, 2), padding='valid')(pad3)
    bn3 = BatchNormalization()(conv3)
    act3 = Activation('relu')(bn3)

    activation = act3
    # attaching 13 residual blocks
    for i in range(13):
        concat = residual_block(activation)
        activation = concat

    conv_1 = Conv2DTranspose(64, kernel_size=(2,2), strides=(2,2), padding='valid')(activation)
    bn_1 = BatchNormalization()(conv_1)
    act_1 = Activation('relu')(bn_1)

    conv_2 = Conv2D(64, kernel_size=(3,3), strides=(1, 1), padding='same')(act_1)
    bn_2 = BatchNormalization()(conv_2)
    act_2 = Activation('relu')(bn_2)

    conv_3 = Conv2D(1, kernel_size=(1,1), strides=(1, 1), padding='same')(act_2)



    model = Model(inputs=x_input, outputs=conv_3)
    model.compile(optimizer='adam', loss=l1_loss.my_loss_l1, metrics=['accuracy'])
    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


mod = create_model(input_size=(128,128,27))