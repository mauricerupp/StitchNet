from utilities import *


def create_very_rand_path(img_path):
    """
    Creates a random path of several unordered and possibly rotated snapshots of one single image
    """
    snaps_per_sample = 5
    snap_size = (64, 64)
    target_size = (128, 128)
    max_step_size = 20

    # read the image
    img_target = cv2.imread(img_path)
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
    # and keeping the aspect ratio
    # since h <= w, we can only check the height
    space = int(2.1*((snaps_per_sample - 1)*max_step_size + snap_size[0]))
    if h <= space:
        img_target = cv2.resize(img_target, (int(w*(space/h)), space)) # (w, h) contrary to all

    # create the stack of image snaps of the target image w/ shape (h, w, 3 * snaps_per_sample)
    img_snaps, covered_pixels, snap_center = create_rand_translation_path(img_target,
                                                                                                 snaps_per_sample,
                                                                                                 snap_size,
                                                                                                 max_step_size)

    # crop the target around the covered area
    snap_top_left = snap_center + np.array(snap_size)/2

    img_target = crop(img_target, snap_top_left, target_size)
    covered_pixels = crop(covered_pixels, snap_top_left, target_size)

    assert img_target.shape[:2] == target_size
    assert covered_pixels.shape[:2] == target_size
    assert img_snaps.shape == (snap_size[0], snap_size[1], 3*snaps_per_sample)

    # save target, snaps and the covered area as numpy arrays and all the paths in one array
    img_target = np.array(img_target)
    covered_pixels = np.array(covered_pixels)

    # revert target to [-1,1]
    img_target = zero_center(img_target/255.0)

    # concatenate target with covered pixels
    img_target = np.concatenate((img_target, covered_pixels), axis=2)

    # revert the snaps to [-1,1] and make a list of length 2
    return [zero_center(img_snaps/255.0), img_target]


def create_rand_translation_path(img_target, snaps_per_sample, snap_size, max_step_size):
    # initialization of local variables
    h_target = img_target.shape[0]
    w_target = img_target.shape[1]
    covered_area = np.zeros([h_target, w_target, 3], dtype=int)
    (h_snap, w_snap) = (snap_size[0], snap_size[1])
    middle_top_left_corner = np.array([0,0])

    for iterationCount in range(snaps_per_sample):
        step_size = random.randint(4, max_step_size) # have a random stepsize for every snapshot
        # update the position of the top left corner of our snap
        if iterationCount == 0: # start in the middle of the image in order to be able to move around
            top_left_corner = np.array([int(h_target/2 - h_snap/2), int(w_target/2 - w_snap)])

        else:
            top_left_corner = update_frame_position(top_left_corner, step_size,
                                                                    h_snap, h_target,
                                                                    w_snap, w_target)

        # update and save the snap
        new_snap = img_target[top_left_corner[0]: top_left_corner[0] + h_snap,
                   top_left_corner[1]: top_left_corner[1] + w_snap]

        # randomly rotate the snap
        new_snap = probability_rotation(new_snap)

        # save the middle frame as center
        middle_top_left_corner += np.array(top_left_corner)

        # update the covered area
        covered_area[top_left_corner[0]: top_left_corner[0] + h_snap,
                     top_left_corner[1]: top_left_corner[1] + w_snap][:] = 1

        assert new_snap.shape == (snap_size[0], snap_size[1], 3)

        # concatenate the new made snap with the "old" snaps
        if iterationCount == 0:
            img_snaps = new_snap
        else:
            img_snaps = np.concatenate((img_snaps, new_snap), axis=2)

    # shuffle the order of the snaps
    img_snaps = shuffle_snaps(img_snaps)

    assert img_snaps.shape == (h_snap, w_snap, 3 * snaps_per_sample) #since there are 3 channels per snap
    return img_snaps, covered_area, middle_top_left_corner/snaps_per_sample


def update_frame_position(topleft_corner, step_size, h_snap, h_target, w_snap, w_target):

    assert topleft_corner[0] >= 0 # since there should be no negative index numbers
    assert topleft_corner[1] >= 0
    assert step_size >= 0
    # have 8 different determined moving directions
    directions = np.array([[0, -1], [1, -1], [1, 0], [1,1], [0,1], [-1, 1], [-1, 0], [-1, -1]])
    # randomly take one direction
    rand = np.random.randint(0,8)

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

    return topleft_corner


def probability_rotation(array):
    return np.rot90(array, random.randint(0, 4))


def shuffle_snaps(snaps_array):
    size = snaps_array.shape[2]
    size = int(size / 3)
    perm = np.random.permutation(size)
    perm *= 3
    shuffled_array = np.zeros(snaps_array.shape)
    for i in range(perm.shape[0]):
        shuffled_array[i*3] = snaps_array[perm[i]]
        shuffled_array[i*3+1] = snaps_array[perm[i]+1]
        shuffled_array[i*3+2] = snaps_array[perm[i]+2]

    return shuffled_array