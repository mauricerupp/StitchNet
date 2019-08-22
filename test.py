import numpy as np
import tensorflow as tf
from losses import l1_loss
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.preprocessing import image
import l1_loss
import l2_loss
import random
from utilities import *
import cv2
from autoencoder_v6 import *

from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.utils import plot_model
import tensorflow as tf
import datetime


input_size=[64,64,3]
img = np.array(cv2.imread('/data/cvg/maurice/logs/ConvAutoencoder_V6_instance_20_80_newcallback/weight_logs/000000000030.jpg'))
autoenc = ConvAutoencoder(input_size, norm='instance', isTraining=False)

img = random_numpy_crop(img, input_size)
y_true = np.expand_dims(img, axis=0)
y_true = np.array(zero_center(y_true/255.0), dtype=np.float32)
y_pred = autoenc.autoencoder(y_true)
y_pred = revert_zero_center(y_pred)*255.0

with tf.Session() as sess:
    #latest = tf.train.latest_checkpoint('/home/maurice/Dokumente/BA/Autoencoder/ConvAutoencoder_V5fixed_instanceBIGGER_20_80_run3/weight_logs/')
    autoenc.load_weights('/data/cvg/maurice/logs/ConvAutoencoder_V6_instance_20_80_newcallback/weight_logs/auto_weights-improvement-20.ckpt')
    y_pred_np = sess.run(y_pred)
    y_pred_np = np.array(np.rint(y_pred_np), dtype=int)

    # save the result
    fig = plt.figure()
    fig.suptitle('Results of predicting', fontsize=20)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('Y_True')
    plt.imshow(img[..., ::-1], interpolation='nearest') # conversion to RGB
    ax3 = fig.add_subplot(1, 2, 2)
    ax3.set_title('Prediction of model')
    plt.imshow(y_pred_np[0][..., ::-1], interpolation='nearest')
    plt.savefig("/data/cvg/maurice/logs/ConvAutoencoder_V6_instance_20_80_newcallback/weight_logs/")
    plt.close()

"""
x = image.load_img('/home/maurice/Dokumente/Try_Models/coco_try/TR/000000504554.jpg')
print(x)
x = np.array(x)
print(x)


directions = np.array([[0, 0], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]])
weights = [0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
print(np.zeros(9))
chosen_dir = np.random.choice(np.arange(0, 9), p=weights)
print(chosen_dir)


directions = np.array([[0,0],[-1, 0], [-1, 1], [0, 1], [1,1], [1,0], [1, -1], [0, -1], [-1, -1]])
print(directions.shape[0])
print(random.randint(1, directions.shape[0]))

img_name = '/home/maurice/Dokumente/Try_Models/coco_try/TR/000000039914.jpg'
test = np.array(cv2.imread(img_name))
print(test.shape)
input_size = (64,64,3)
te = random_numpy_crop(test, input_size)
print(te.shape)

img_name = '/home/maurice/Dokumente/Try_Models/coco_try/TR/000000039914.jpg'
print(img_name)
test = np.array(cv2.imread(img_name))
test2 = np.array(cv2.imread(img_name))
te = np.concatenate([test, test2], axis=2)
size = te.shape


input_size = (64,64,3)
img = random_numpy_crop(test, input_size)
y_true = np.expand_dims(img, axis=0)
conc = np.concatenate([y_true, y_true], axis=3)
testmod = ConvAutoencoder(input_size)
testmod.load_encoder_weights('/home/maurice/Dokumente/encoder_logs/')
index = 1
in_conv_list = []
input_size = (64, 64, 6)
for i in range(0, 3, 3):
    x = Lambda(lambda x: x[:, :, :, i:i + 3], name='img_{}'.format(str(index)))(conc)
    in_conv_list.append(testmod.encoder.predict(x))
    index += 1

print(in_conv_list[0].shape)



size1 = np.array([488, 488,3])
size2 = np.array([10,10,3])

img_name = '/home/maurice/Dokumente/Try_Models/coco_try/TR/000000039914.jpg'
test = np.array(cv2.imread(img_name))
print(test.shape)
assert test.shape[2] == size1[2]
print(random_numpy_crop(test, size1).shape)


batchlist = []
batchlist.append(img_name)
batchlist.append(img_name)
img_size = np.array([64,64,3])
test2 = random_numpy_crop(zero_center(np.array(cv2.imread(img_name))/255.0), img_size)
test3 =random_numpy_crop(zero_center(np.array(cv2.imread(img_name))/255.0), img_size)
print(type(test2))
print(test2.shape)
test = np.stack([random_numpy_crop(zero_center(np.array(cv2.imread(img))/255.0), img_size) for img in batchlist], axis=0)
#sess = tf.InteractiveSession()
#with sess.as_default():
#    test = test.eval()
print(type(test))
print(test.shape)

y_pred1 = np.array([[[0, 255, 2],[33, 22, 11]], [[0, 0, 222],[1, 0, 2]]])
print(y_pred1)
print(y_pred1.shape)
y_pred1 = preprocess_input(y_pred1, mode='torch')
print(y_pred1)
print(y_pred1.shape)

def create_model(pretrained_weights=None, input_size=None, G0=64, G=32, D=20, C=6):
    
    A simple residual network with ResBlocks from Givi
    :param G0: filtersize for the last convolutional layer
    :param G: filtersize per convolutional layer
    :param D: amout of residual dense blocks (RDB)
    :param C: Amount of convolutional layers per RDB
    :param pretrained_weights:
    :param input_size:
    :return:
    inputs = Input(input_size)
    conv = Conv2D(128, kernel_size=3, activation='relu', padding='same', name='global_conv2')(inputs)
    conv = create_resblock(conv, 'dx', 128, 3, 1, 1, BatchNormalization)
    model = Model(inputs=inputs, outputs=conv)
    model.compile(optimizer='adam', loss=l1_loss.custom_loss, metrics=['accuracy'])
    model.summary()


def create_resblock(prior_layer, block_name, n_filters, kernel_size, stride, dilation, normalizer):
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, padding='same', name='global_conv3')(prior_layer)
    x = normalizer()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=n_filters, kernel_size=kernel_size,
               strides=stride, dilation_rate=dilation, name=block_name + "conv2")(x)

    x = normalizer()(x)

    return x
    #return Add()([prior_layer, x])


mod = create_model(input_size=(64,64,15))

y_true = np.load('/home/maurice/Dokumente/Try_Models/coco_try/train/targets/target1.npy')[:,:,-3:]
print(y_true.shape)
y_true = np.concatenate([y_true[:,:,:-3], y_true[:,:,-3:]], axis=2)
print(y_true.shape)


y_true1 = np.array([[[0, 0.2, 0.1, 1,1,1],[0.4, 0.2, 0.1,0,0,0]], [[0.3, 0.0, 0.5,1,1,1],[0.4, 0.4, 0.1,0,0,0]]])
y_true2 = np.array([[[0.11, 0.22, 0.33, 0,0,0],[0.33, 0.22, 0.11,0,0,0]], [[0.7, 0.70, 0.0,0,0,0],[0.1, 0.0, 0.2,1,1,1]]])
y_true = np.array((y_true1, y_true2))
y_pred1 = np.array([[[-0.5, 0.2, 0.1],[0.330, 0.220, 0.110]], [[0.3, 1.0, 0.5],[0.1, 0.0, 0.20]]])
y_pred2 = np.array([[[0.11, 0.22, 0.33],[0.33, 0.22, 0.11]], [[0.7, 0.70, 0.0],[0.1, 0.0, 0.2]]])
y_pred = np.array(((y_pred1, y_pred2)))


sess = tf.InteractiveSession()
#loss = l1_loss.custom_loss(y_true, y_pred)
loss = l2_loss.my_loss_l2(y_true, y_pred)
equality = tf.equal(y_pred1, y_pred2 / 255.0)
accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

print(sess.run(loss))
sess.close()

equality = np.equal(y_pred1, y_pred2 / 255.0)
accuracy = np.mean(equality)

print(accuracy)
print('Results of predicting Image {} on epoch {} with an accuracy of {:.2%}'.format(20, 1 + 1, accuracy))

y_true1 = np.array([[[0, 2, 1, 1,1,1],[4, 2, 1,0,0,0]], [[3, 0, 5,1,1,1],[4, 4, 1,0,0,0]]])
print(y_true1.shape)
y_true1 = y_true1[0:1, 0:1]
print(y_true1.shape)
print(y_true1)
print(4 * (100, 150))
"""
