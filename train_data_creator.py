import numpy as np
import os
import cv2
import random as ran
import math
import matplotlib.pyplot as plt

# Set the constants
# where we would output the sample images if needed:
TRAINDIR = '/home/maurice/Dokumente/Try_Models/coco_try/test'

os.environ['CUDA_VISIBLE_DEVICES'] = str(1)


# initialize the global variables
sample_count = 0
coverage = 0
overlapse = 0


# creates the training data and stores every sample as numpy array in target_dir and snap_dir
# targets consists of 6 channels :
# the first 3 are the color channels and the 2nd 3 are whether these pixels are covered or not
def create_training_data(raw_dir, target_dir, snap_dir, paths_dir, target_size, snap_size, snaps_per_sample):
    assert target_size > snap_size
    assert snaps_per_sample > 0
    snaps_paths = []
    targets_paths = []

    for img in os.listdir(raw_dir):
        global coverage
        global sample_count
        global overlapse

        # read the image
        img_target = cv2.imread(os.path.join(raw_dir, img))
        (h, w) = img_target.shape[:2]
        
        # if the image is upright, turn it by 90 degrees
        if h > w:
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, 90, 1)
            img_target = cv2.warpAffine(img_target, M, (h, w))

        # reshape the target to a size where we have enough space to move around
        img_target = resize_img(img_target, (4*snap_size[0], 4*snap_size[1]))

        # create the stack of image snaps of the target image w/ shape (h, w, 3 * snaps_per_sample)
        img_snaps, covered_pixels, img_overlapse, corner_list = create_snap_path_translation(img_target, snaps_per_sample, snap_size)
        sample_count += 1

        # crop the target around the covered area
        h_center = int(corner_list[0] / snaps_per_sample + snap_size[0] / 2)
        w_center = int(corner_list[1] / snaps_per_sample + snap_size[1] / 2)
        center_corner = (h_center, w_center)
        #plt.imshow(img_overlapse)
        #plt.show()
        img_target = crop(img_target, center_corner, target_size)
        covered_pixels = crop(covered_pixels, center_corner, target_size)
        img_overlapse = crop(img_overlapse, center_corner, target_size)
        #plt.imshow(img_overlapse)
        #plt.show()
        
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

        #plt.imshow(img_overlapse)
        #plt.show()

        # update the overall coverage
        coverage += np.count_nonzero(covered_pixels) / covered_pixels.size

        # update the overall overlapse
        overlapse += (np.count_nonzero(img_overlapse !=1)-np.count_nonzero(img_overlapse == 0)) / img_overlapse.size

    # save the paths as numpy arrays
    np.save(paths_dir + "/targets_paths", targets_paths)
    np.save(paths_dir + "/snaps_paths", snaps_paths)

    print("Coverage of the created dataset is {:.2%}".format(coverage / sample_count))
    print("Overlapse of the created dataset is {:.2%}".format(overlapse / sample_count))
    print("Your data is stored in:" + str(raw_dir) + "and " + str(snap_dir))


def create_snap_path_translation(img_target, snaps_per_sample, snap_size):

    # initialization of local variables
    top_left_corners_list = []
    h_target = img_target.shape[0]
    w_target = img_target.shape[1]
    covered_area = np.zeros([h_target, w_target, 3], dtype=int)
    img_overlapse = np.zeros([h_target, w_target], dtype=int)
    (h_snap, w_snap) = (snap_size[0], snap_size[1])

    # initialize the top left corner of the first snap roughly in the top middle
    top_left_corner = np.array([ran.randint(0, int(h_target / 3)), ran.randint(w_snap, int(w_target / 2))])
    img_snaps = img_target[top_left_corner[0]: top_left_corner[0] + h_snap,
               top_left_corner[1]: top_left_corner[1] + w_snap]
    angle = 0
    top_left_corners_list = top_left_corner.copy()

    for iterationCount in range(snaps_per_sample -1):
        # update the covered area
        covered_area[top_left_corner[0]: top_left_corner[0] + h_snap,
                     top_left_corner[1]: top_left_corner[1] + w_snap][:] = 1

        # update the overlapse
        img_overlapse[top_left_corner[0]: top_left_corner[0] + h_snap,
                     top_left_corner[1]: top_left_corner[1] + w_snap] += 1

        # update the angle
        angle += ran.uniform(math.pi / 5, math.pi / 3)

        # update the position of the top left corner of our snap
        top_left_corner, angle = update_frame_position(top_left_corner, angle, h_snap, h_target, w_snap, w_target)
        top_left_corners_list += top_left_corner
        # update and save the snap
        new_snap = img_target[top_left_corner[0]: top_left_corner[0] + h_snap,
                    top_left_corner[1]: top_left_corner[1] + w_snap]
        assert new_snap.shape == (snap_size[0], snap_size[1], 3)

        """
        # save the frame as an image (debugging)
        cv2.imwrite(os.path.join(TRAINDIR, 'pic{}-itr{}.jpeg'.format(sample_count, iterationCount)), new_snap)
        """

        # concatenate the new made snap with the "old" snaps
        img_snaps = np.concatenate((img_snaps, new_snap), axis=2)

        #plt.imshow(img_overlapse)
        #plt.show()
    assert img_snaps.shape == (h_snap, w_snap, 3 * snaps_per_sample) #since there are 3 channels per snap
    return img_snaps, covered_area, img_overlapse, top_left_corners_list


def update_frame_position(topleft_corner, angle, h_snap, h_target, w_snap, w_target):

    assert topleft_corner[0] >= 0 # since there should be no negative index numbers
    assert topleft_corner[1] >= 0

    # create the stepsize according to our snap size in order to have overlapping images
    # make it a new value for every "camera move" to simulate a non-constant speed
    step_size = min(w_snap, h_snap) / ran.uniform(1.001, 1.2)

    # check if the new frame would be completely inside the picture
    if 0 < topleft_corner[0] + round(step_size * math.sin(angle)) < h_target - h_snap \
            and 0 < topleft_corner[1] + round(step_size * math.cos(angle)) < w_target - w_snap:
        pass
    # if we would hit a border we simply try all 4 "normal" directions
    else:
        if 0 < topleft_corner[0] + round(step_size * math.sin(math.pi/2)) < h_target - h_snap \
                and 0 < topleft_corner[1] + round(step_size * math.cos(math.pi/2)) < w_target - w_snap:
            angle = math.pi/2
        elif 0 < topleft_corner[0] + round(step_size * math.sin(math.pi)) < h_target - h_snap \
                and 0 < topleft_corner[1] + round(step_size * math.cos(math.pi)) < w_target - w_snap:
            angle = math.pi
        elif 0 < topleft_corner[0] + round(step_size * math.sin(1.5 * math.pi)) < h_target - h_snap \
                and 0 < topleft_corner[1] + round(step_size * math.cos(1.5 * math.pi)) < w_target - w_snap:
            angle = 1.5 * math.pi
        elif 0 < topleft_corner[0] + round(step_size * math.sin(0)) < h_target - h_snap \
                and 0 < topleft_corner[1] + round(step_size * math.cos(0)) < w_target - w_snap:
            angle = 0
        else:
            step_size = 0
            angle = ran.uniform(0, 2 * math.pi)

    topleft_corner[0] = topleft_corner[0] + round(step_size * math.sin(angle))
    topleft_corner[1] = topleft_corner[1] + round(step_size * math.cos(angle))

    return topleft_corner, (angle % (2 * math.pi))


def resize_img(img, desired_size):
    # scale the image until it covers the whole desired size
    img = scale_img(img, desired_size)

    (h, w) = img.shape[:2]
    # cut the image so that the cut frame is centered
    h_cut = h - desired_size[0]
    space_h = int(h_cut / 2)
    w_cut = w - desired_size[1]
    space_w = int(w_cut / 2)

    assert h_cut >= 0
    assert w_cut >= 0

    img = img[space_h:space_h+desired_size[0], space_w:space_w+desired_size[1]]

    return img


def scale_img(img, des_size):
    old_size = img.shape[:2]
    h_ratio = des_size[0] / old_size[0]
    w_ratio = des_size[1] / old_size[1]

    # find out how to scale it, so that it still covers the whole area
    if h_ratio * old_size[1] > des_size[1]:
        ratio = h_ratio
    else:
        ratio = w_ratio

    # resize the image while keeping its ratio so its not warped
    new_size = tuple([round(x*ratio) for x in old_size])
    return cv2.resize(img, (new_size[1], new_size[0]))

def crop(img, center, size):
    assert center[0] >= 0
    assert center[1] >= 0
    (h, w) = img.shape[:2]
    (up, under, left, right) = (center[0]-size[0] // 2, center[0] + size[0] - (size[0] // 2),
                                center[1] - size[1] // 2, center[1] + size[1] - (size[1] // 2))
    if up >= 0:
        if left >= 0:
            pass
        else:
            left = 0
            right = size[1]
    else:
        up = 0
        under = size[0]
        if left >= 0:
            pass
        else:
            left = 0
            right = size[1]
    if under <= h:
        if right <= w:
            pass
        else:
            right = w
            left = w-right
    else:
        under = h
        up = h - under
        if right <= w:
            pass
        else:
            right = w
            left = w-right

    img = img[up:under, left:right]
    assert img.shape[:2] == (size[0], size[1])
    return img


create_training_data('/home/maurice/Dokumente/Try_Models/coco_try/RAW_train',
                     '/home/maurice/Dokumente/Try_Models/coco_try/train/targets',
                     '/home/maurice/Dokumente/Try_Models/coco_try/train/snaps',
                     '/home/maurice/Dokumente/Try_Models/coco_try/train/',
                     (300, 450), (100, 150), 8)
"""
create_training_data('/data/cvg/maurice/unprocessed/coco_val',
                     '/data/cvg/maurice/processed/coco/val/targets',
                     '/data/cvg/maurice/processed/coco/val/snaps',
                     '/data/cvg/maurice/processed/coco/val/',
                     (400, 600), (100, 150), 8)
"""
