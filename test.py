import numpy as np
import matplotlib.pyplot as plt


target = np.load('/home/maurice/Dokumente/Try_Models/coco_try/train/targets/target{}.npy'.format(5))
print(target.shape)
target = target /255.0
print(target)
target = target * 255.0
print(target)
x = np.array([[10,50,30], [20,30,40]])
y = np.array([20,30,40])

x2 = np.array([100,20,10])
y2 = np.array([20,70,60])

plt.plot(x, marker="o", linestyle='-')
plt.show()
"""
# load Y
target = np.load('/home/maurice/Dokumente/Try_Models/coco_try/train/targets/target{}.npy'.format(5))
target = target[:, :, :-3]
# display the result
fig = plt.figure()
fig.suptitle('Results Of Prediction', fontsize=20)
ax1 = fig.add_subplot(1,3,1)
ax1.set_title('Y_True')
plt.imshow(target, interpolation='nearest')
ax2 = fig.add_subplot(1,3,2)
ax2.set_title('Y_True covered')
plt.imshow(target, interpolation='nearest')
ax3 = fig.add_subplot(1,3,3)
ax3.set_title('Prediction of model')
plt.imshow(target, interpolation='nearest')
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.savefig("/home/maurice/test.png")
plt.show()
"""

"""
y_true1 = np.array([[[0, 2, 1, 1,1,1],[4, 2, 1,0,0,0]], [[3, 0, 5,1,1,1],[4, 4, 1,0,0,0]]])
y_true2 = np.array([[[11, 22, 33, 0,0,0],[33, 22, 11,0,0,0]], [[7, 70, 0,0,0,0],[1, 0, 2,1,1,1]]])
y_true = np.array((y_true1, y_true2))
y_pred1 = np.array([[[1, 2, 1],[330, 220, 110]], [[3, 0, 5],[1, 0, 20]]])
y_pred2 = np.array([[[100, 200, 100],[330, 220, 110]], [[3, 0, 5],[0, 1, 2]]])
y_pred = np.array(((y_pred1, y_pred2)))

sess = tf.InteractiveSession()
loss = loss.my_loss_l1(y_true, y_pred)
print(sess.run(loss))
print(loss.eval())
sess.close()




y_true1 = np.array([[[0, 2, 1, 1,1,1],[4, 2, 1,0,0,0]], [[3, 0, 5,1,1,1],[4, 4, 1,0,0,0]]])
print(y_true1.shape)
y_true1 = y_true1[0:1, 0:1]
print(y_true1.shape)
print(y_true1)
print(4 * (100, 150))
np.load()
"""
