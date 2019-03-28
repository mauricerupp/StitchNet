import numpy as np
import os
import cv2
import random as ran
import math

# Set the constants
# where we would output the sample images if needed:
TRAINDIR = '/home/maurice/Dokumente/Try_Models/coco_try/TR'
# amount of frames we take per picture:


# initialize the global variables
sample_count = 0
coverage = 0


def create_training_data(DATADIR, PICAMOUNTPEREXAMPLE, FRAMERATIO):
    training_data = []
    max_dim = [0, 0]

    for img in os.listdir(DATADIR):
        global coverage

        # read every image
        imgArray_y = cv2.imread(os.path.join(DATADIR, img))
        (h, w) = imgArray_y.shape[:2]
        if h > w: #if the image is upright, turn it 90 degrees
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, 90, 1)
            imgArray_y = cv2.warpAffine(imgArray_y, M, (h, w))

        # update our maximal dimension for padding
        if h > max_dim[0]:
            max_dim[0] = h
        if w > max_dim[1]:
            max_dim[1] = w

        imgArray_x, covered_pixels = create_image_path_translation(imgArray_y, PICAMOUNTPEREXAMPLE, FRAMERATIO)

        # add the image to our training data and update the coverage
        training_data.append([imgArray_x, imgArray_y, covered_pixels])
        coverage += np.count_nonzero(covered_pixels) / covered_pixels.size
    print("Coverage of the created dataset is {:.2%}".format(coverage / sample_count))
    return training_data, max_dim


def create_image_path_translation(rawImageArray, PICAMOUNTPEREXAMPLE, FRAMERATIO):
    # initialization of local variables
    global sample_count
    imgList = []
    height = rawImageArray.shape[0]
    width = rawImageArray.shape[1]
    covered_area = np.zeros([height, width], dtype=int)
    # create the framesize of our screenshots
    frame_height = int(height / ran.randint(2, 4))
    frame_width = int(frame_height * FRAMERATIO)
    # initialize the top left corner of the first frame
    top_left_corner = np.array([ran.randint(0, height-frame_height), ran.randint(0, width-frame_width)])

    # initialize the original movement angle randomly
    angle = ran.uniform(0, 2 * math.pi)
    for iterationCount in range(PICAMOUNTPEREXAMPLE):
        # update the angle in a "motion" between -90 and 90 degrees in order to simulate a camera move
        angle += ran.uniform(-math.pi / 2, math.pi / 2)

        # update the position of the top left corner of our frame
        top_left_corner, angle = update_frame_position(top_left_corner, angle, frame_height, height, frame_width, width)

        new_frame = rawImageArray[top_left_corner[0]: top_left_corner[0] + frame_height,
                    top_left_corner[1]: top_left_corner[1] + frame_width]
        assert new_frame.shape == (frame_height, frame_width, 3)

        """
        # save the frame as an image (debugging)
        cv2.imwrite(os.path.join(TRAINDIR, 'pic{}-itr{}.jpeg'.format(sample_count, iterationCount)), new_frame)
        """

        # append the frame to the list
        imgList.append(new_frame)

        # update the covered area
        covered_area[top_left_corner[0]: top_left_corner[0] + frame_height,
                     top_left_corner[1]: top_left_corner[1] + frame_width] = 1

    sample_count += 1
    assert len(imgList) == PICAMOUNTPEREXAMPLE
    return stack_images(imgList), covered_area


def update_frame_position(topleft_corner, angle, frame_height, height, frame_width, width):

    assert topleft_corner[0] >= 0 # since there should be no negative index numbers
    assert topleft_corner[1] >= 0
    # create the stepsize according to our framesize in order to have overlapping images
    # make it a new value for every "cameramove" to simulate a non-constant speed
    stepsize = min(frame_width, frame_height) / ran.uniform(1.001, 2)

    # check if the new frame would be completely inside the picture
    if 0 < topleft_corner[0] + round(stepsize * math.sin(angle)) < height - frame_height \
            and 0 < topleft_corner[1] + round(stepsize * math.cos(angle)) < width - frame_width:
        topleft_corner[0] = topleft_corner[0] + round(stepsize * math.sin(angle))
        topleft_corner[1] = topleft_corner[1] + round(stepsize * math.cos(angle))

    # if we would hit a border frame, we simply take a small step to the middle to move away from the border
    else:
        middle = [round(height/2 - frame_height/2), round(width/2 - frame_width / 2)] #these are the coordinates of the topleft corner for a centered frame
        vect = [middle[0] - topleft_corner[0], middle[1] - topleft_corner[1]]  # to get the direction where we want to move
        if vect == [0, 0]:  # avoid the case where the topleft corner is exactly in the middle
            vect = [0.1, 0.1]
        vect = vect / np.sum(np.absolute(vect))  # in order to scale the direction vector to sum up to 1
        topleft_corner[0] = topleft_corner[0] + round(20 * vect[0]) # 2 is the stepsize of how far we go to to the middle
        topleft_corner[1] = topleft_corner[1] + round(20 * vect[1])
    return topleft_corner, (angle % (2 * math.pi))


#stack the images into a x*x 2D image
def stack_images(image_list):
    img_per_row = int(math.sqrt(len(image_list)))

    h_stack = []
    row_num = 0
    for i in range(img_per_row):
        h_stack.append(np.hstack(image_list[row_num: row_num+img_per_row]))
        row_num += img_per_row

    return np.vstack(h_stack)

