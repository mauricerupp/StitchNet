import numpy as np
import os
import cv2
import random as ran
import math
import matplotlib.pyplot as plt
import random
from utilities import *


# Set the constants
# where we would output the sample images if needed:
TRAINDIR = '/home/maurice/Dokumente/Try_Models/coco_try/test'

#os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

# initialize the global variables
sample_count = 0
coverage = 0
overlapse = 0


def create_training_data(raw_dir, target_dir, snap_dir, paths_dir, target_size, snap_size, snaps_per_sample):
    """
    creates the training data and stores every sample as numpy array in target_dir and snap_dir
    targets consists of 6 channels:
    the first 3 are the color channels and the 2nd 3 are whether these pixels are covered or not
    creates a random path, where the center of the frame is the middle snap, also the paths are more camera-like
    so there are different probabilities for going in the same direction as before as for going back
    :param raw_dir:
    :param target_dir:
    :param snap_dir:
    :param paths_dir:
    :param target_size:
    :param snap_size:
    :param snaps_per_sample:
    :param step_size:
    """
    assert target_size > snap_size
    assert snaps_per_sample > 0
    snaps_paths = []
    targets_paths = []
    global coverage
    global sample_count
    global overlapse
    sample_count = 0
    coverage = 0
    overlapse = 0

    for img in os.listdir(raw_dir):
        # read the image
        img_target = cv2.imread(os.path.join(raw_dir, img))
        (h, w) = img_target.shape[:2]

        # if the image is upright, turn it by 90 degrees
        if h >= w:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, 90, 1)
            cos = np.abs(M[0,0])
            sin = np.abs(M[0, 1])
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))
            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nW / 2) - center[0]
            M[1, 2] += (nH / 2) - center[1]
            img_target = cv2.warpAffine(img_target, M, (h, w))
            (h, w) = img_target.shape[:2]
            assert h <= w

        # reshape the target to a size where we have enough space to move around if its to small
        # and keep the aspect ratio
        # since h <= w, we can only adjust it with h
        space = int(2.4*((snaps_per_sample - 1)*step_size + snap_size[0]))
        if h <= space:
            img_target = cv2.resize(img_target, (int(w*(space/h)), space)) # (w, h) contrary to all

        # create the stack of image snaps of the target image w/ shape (h, w, 3 * snaps_per_sample)
        img_snaps, covered_pixels, img_overlapse, middle_frame_top_left_corner = create_rand_translation_path(img_target,
                                                                                                     snaps_per_sample,
                                                                                                     snap_size)
        sample_count += 1
        middle_frame_center = middle_frame_top_left_corner + np.array(snap_size)/2
        # crop the target around the covered area
        #plt.imshow(img_overlapse)
        #plt.show()
        img_target = crop(img_target, middle_frame_center, target_size)
        covered_pixels = crop(covered_pixels, middle_frame_center, target_size)
        img_overlapse = crop(img_overlapse, middle_frame_center, target_size)
        #plt.imshow(img_overlapse)
        #plt.show()

        """
        # save the frame as an image (debugging)
        cv2.imwrite(os.path.join(TRAINDIR, 'pic{}-.jpeg'.format(sample_count)), img_target)
        """
        assert img_target.shape[:2] == target_size
        assert covered_pixels.shape[:2] == target_size
        assert img_snaps.shape == (snap_size[0], snap_size[1], 3*snaps_per_sample)
        # save target, snaps and the covered area as numpy arrays and all the paths in one array
        img_target = np.array(img_target)
        covered_pixels = np.array(covered_pixels)
        #plt.imshow(covered_pixels[:,:,0])
        #plt.show()

        img_target = np.concatenate((img_target, covered_pixels), axis=2)
        np.save(target_dir + "/" + "target" + str(sample_count), img_target)
        targets_paths.append(target_dir + "/" + "target" + str(sample_count) + ".npy")

        img_snaps = np.array(img_snaps)
        np.save(snap_dir + "/" + "snaps" + str(sample_count), img_snaps)
        snaps_paths.append(snap_dir + "/" + "snaps" + str(sample_count) + ".npy")

        # update the overall coverage
        coverage += np.count_nonzero(covered_pixels) / covered_pixels.size

        # update the overall overlapse
        overlapse += (np.count_nonzero(img_overlapse !=1)-np.count_nonzero(img_overlapse == 0)) / np.count_nonzero(img_overlapse)
        temp = img_target[:, :, :-3]
        fig5 = plt.imshow(temp[..., ::-1])
        plt.savefig(
            "/home/maurice/Dokumente/BA/Docu_smooth_random_samples/img{}-target-cropped.png".format(sample_count - 1))

        temp = covered_pixels * temp
        fig7 = plt.imshow(temp[..., ::-1])
        plt.savefig("/home/maurice/Dokumente/BA/Docu_smooth_random_samples/img{}-coveredtarget-cropped.png".format(
            sample_count - 1))

        fig8 = plt.imshow(img_overlapse)
        plt.savefig("/home/maurice/Dokumente/BA/Docu_smooth_random_samples/img{}-cropped.png".format(sample_count - 1))

        img_target = None
        covered_pixels = None
        img_snaps = None
        img_overlapse = None

    # save the paths as numpy arrays
    np.save(paths_dir + "/targets_paths", targets_paths)
    np.save(paths_dir + "/snaps_paths", snaps_paths)

    print("Coverage of the created dataset is {:.2%}".format(coverage / sample_count))
    print("Overlapse of the created dataset is {:.2%}".format(overlapse / sample_count))
    print("Your data is stored in:" + str(target_dir) + " and " + str(snap_dir))


def create_rand_translation_path(img_target, snaps_per_sample, snap_size):

    # initialization of local variables
    h_target = img_target.shape[0]
    w_target = img_target.shape[1]
    covered_area = np.zeros([h_target, w_target, 3], dtype=int)
    img_overlapse = np.zeros([h_target, w_target], dtype=int)
    (h_snap, w_snap) = (snap_size[0], snap_size[1])
    middle_frame_top_left_corner = []

    for iterationCount in range(snaps_per_sample):
        last_direction = np.random.randint(0, 8) #completely random direction to start with
        # update the position of the top left corner of our snap
        if iterationCount == 0: # start in the middle of the image in order to be able to move around
            top_left_corner = np.array([int(h_target/2 - h_snap/2), int(w_target/2 - w_snap)])

        else:
            top_left_corner, last_direction = update_frame_position(top_left_corner,
                                                                    h_snap, h_target,
                                                                    w_snap, w_target, last_direction)


        # update and save the snap
        new_snap = img_target[top_left_corner[0]: top_left_corner[0] + h_snap,
                   top_left_corner[1]: top_left_corner[1] + w_snap]

        # save the middle frame as center
        if iterationCount == int(snaps_per_sample/2):
            middle_frame_top_left_corner = np.array(top_left_corner)

        # update the covered area
        covered_area[top_left_corner[0]: top_left_corner[0] + h_snap,
                     top_left_corner[1]: top_left_corner[1] + w_snap][:] = 1

        # update the overlapse
        img_overlapse[top_left_corner[0]: top_left_corner[0] + h_snap,
                     top_left_corner[1]: top_left_corner[1] + w_snap] += 1

        #plt.imshow(img_overlapse)
        #plt.show()

        assert new_snap.shape == (snap_size[0], snap_size[1], 3)

        """
        # save the frame as an image (debugging)
        cv2.imwrite(os.path.join(TRAINDIR, 'pic{}-itr{}.jpeg'.format(sample_count, iterationCount)), new_snap)
        """

        # concatenate the new made snap with the "old" snaps
        if iterationCount == 0:
            img_snaps = new_snap
        else:
            img_snaps = np.concatenate((img_snaps, new_snap), axis=2)

    assert img_snaps.shape == (h_snap, w_snap, 3 * snaps_per_sample) #since there are 3 channels per snap
    return img_snaps, covered_area, img_overlapse, middle_frame_top_left_corner


def update_frame_position(topleft_corner, h_snap, h_target, w_snap, w_target, last_direction):

    assert topleft_corner[0] >= 0 # since there should be no negative index numbers
    assert topleft_corner[1] >= 0
    # have 8 different determined moving directions
    directions = np.array([[0, -1], [1, -1], [1, 0], [1,1], [0,1], [-1, 1], [-1, 0], [-1, -1]])
    weights = fill_probabilities(last_direction)
    step_size = random.randint(5, 20)

    rand = np.random.choice(np.arange(0, 8), p=weights)

    # check if the new frame would be completely inside the picture
    if 0 < topleft_corner[0] + directions[rand][0]*step_size < h_target - h_snap \
            and 0 < topleft_corner[1] + directions[rand][1]*step_size < w_target - w_snap:
        pass

    # since this case should never occur
    else:
        print("out of borders")
        exit()

    topleft_corner[0] = topleft_corner[0] + directions[rand][0]*step_size
    topleft_corner[1] = topleft_corner[1] + directions[rand][1]*step_size

    return topleft_corner, rand


def opp_dir(direction):
    if direction >= 4:
        return direction - 4
    else:
        return direction + 4


def fill_probabilities(previous_direction):
    weights = np.zeros(8)
    weights[previous_direction] = 0.33
    # adust the weight left of the prev direction
    if previous_direction > 0:
        weights[previous_direction-1] = 0.23
        weights[opp_dir(previous_direction-1)] = 0.02
    else:
        weights[7] = 0.23
        weights[opp_dir(7)] = 0.02
    # adjust the weight right of the prev direction
    if previous_direction < 7:
        weights[previous_direction+1] = 0.23
        weights[opp_dir(previous_direction+1)] = 0.02
    else:
        weights[0] = 0.23
        weights[opp_dir(0)] = 0.02
    # adjust the weights at a right angle to the previous direction
    if previous_direction < 6:
        weights[previous_direction+2] = 0.085
        weights[opp_dir(previous_direction+2)] = 0.085
    else:
        weights[previous_direction-2] = 0.085
        weights[opp_dir(previous_direction - 2)] = 0.085
    return weights


create_training_data('/home/maurice/Dokumente/Try_Models/coco_try/TR',
                     '/home/maurice/Dokumente/Try_Models/coco_try/train/targets',
                     '/home/maurice/Dokumente/Try_Models/coco_try/train/snaps',
                     '/home/maurice/Dokumente/Try_Models/coco_try/train',
                     (128, 128), (64, 64), 5)
"""
create_training_data('/home/maurice/Dokumente/Try_Models/coco_try/TR',
                     '/home/maurice/Dokumente/Try_Models/coco_try/val/targets',
                     '/home/maurice/Dokumente/Try_Models/coco_try/val/snaps',
                     '/home/maurice/Dokumente/Try_Models/coco_try/val',
                     (128, 128), (64, 64), 5, 16)



create_training_data('/data/cvg/maurice/unprocessed/coco_train',
                     '/data/cvg/maurice/processed/coco/train/targets',
                     '/data/cvg/maurice/processed/coco/train/snaps',
                     '/data/cvg/maurice/processed/coco/train/',
                     (128, 128), (64, 64), 5, 16)

create_training_data('/data/cvg/maurice/unprocessed/coco_val',
                     '/data/cvg/maurice/processed/coco/val/targets',
                     '/data/cvg/maurice/processed/coco/val/snaps',
                     '/data/cvg/maurice/processed/coco/val/',
                     (128, 128), (64, 64), 5, 16)
"""
