import numpy as np
import u_net_convtrans_model2
import scipy.misc

model = u_net_convtrans_model2.create_model(pretrained_weights='/home/maurice/Dokumente/'
                                                               'BA-Models_logs/u-net-convtrans-model2/weight_logs/',
                                            input_size=(128,128,27))
tester = np.load('/home/maurice/Dokumente/Try_Models/coco_try/train/snaps/snaps1.npy')
tester = np.expand_dims(tester, axis=0)

target = np.load('/home/maurice/Dokumente/Try_Models/coco_try/train/targets/target1.npy')
rgb = scipy.misc.toimage(target[:, :, :-3])
scipy.misc.imshow(rgb)

pred = model.predict(tester)
rgb = scipy.misc.toimage(pred[0, :, :, :])
scipy.misc.imshow(rgb)