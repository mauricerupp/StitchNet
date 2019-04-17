import numpy as np
import u_net_convtrans_model3
from matplotlib import pyplot as plt

model_name = u_net_convtrans_model3

model = model_name.create_model(pretrained_weights=
                                '/home/maurice/Dokumente/BA-Models_logs/unshuffled_samples/{}/weight_logs/'.format(model_name.__name__),
                                input_size=(128,128,27))

for i in range(1,20):
    # load X
    tester = np.load('/home/maurice/Dokumente/Try_Models/coco_try/train/snaps/snaps{}.npy'.format(i))
    tester = np.expand_dims(tester, axis=0)

    # load Y
    target = np.load('/home/maurice/Dokumente/Try_Models/coco_try/train/targets/target{}.npy'.format(i))
    covered_area = target[:, :, -3:]
    target = target[:, :, :-3]
    covered_target = target * covered_area

    # predict Y
    pred = model.predict(tester)
    pred = np.array(np.rint(pred), dtype=int)

    # display the result
    fig = plt.figure()
    fig.suptitle('Results Of Prediction', fontsize=20)
    ax1 = fig.add_subplot(1,3,1)
    ax1.set_title('Y_True')
    plt.imshow(target, interpolation='nearest')
    ax2 = fig.add_subplot(1,3,2)
    ax2.set_title('Y_True covered')
    plt.imshow(covered_target, interpolation='nearest')
    ax3 = fig.add_subplot(1,3,3)
    ax3.set_title('Prediction of model')
    plt.imshow(pred[0], interpolation='nearest')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
