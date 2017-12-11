import matplotlib.pyplot as plt
import cv2
import os
import random
import math
import shutil
import numpy as np
from xml.etree.ElementTree import Element, SubElement, ElementTree, dump

'''
Input:
    - character image files classified with different folders
        (ex: image files of phd08 dataset)
    - background image files
        (ex: DTD dataset, https://www.robots.ox.ac.uk/~vgg/data/dtd/)

Output:
    - synthetic images (character images patched on background images)
    - annotation (xml with labels, positions and sizes)
    - txt files (train / validation / test set splitted)

Usage:
    1) Download phd08 dataset (https://drive.google.com/drive/folders/0B-u3H0N7Z4OueFgtRDZCRmtmeFk)
    2) Download background image dataset
    3) Download phd08_to_image.py (https://gitlab2.kabang.io/bigdata/ocr/tree/master/preprocessing/phd08_to_tfrecord)
    4) Run 'phd08_to_image.py' to convert txt files to image files => Output: phd08_out
    5) Configure parameters of 'synthetic_text_image.py'
    6) Run 'synthetic_text_image.py'
'''

# paramters
# ---------------------------------------------------------------
# character image files
SOURCE_CHAR_DIR = "/Users/asher/dev/workspace/exercise/phd08_to_tfrecord/phd08_out"
SOURCE_CHAR_FORMAT = '.jpeg'

# background image files
SOURCE_BG_DIR = "/Users/asher/dev/workspace/exercise/cv/background"
SOURCE_BG_FORMAT = '.jpg'

# target path/format
TARGET_ROOT_DIR = "synth_out"
TARGET_IMG_DIR = TARGET_ROOT_DIR + "/images"
TARGET_ANNO_DIR = TARGET_ROOT_DIR + "/annotation"
TARGET_FILE_FORMAT = '.jpg'

# the number of output (synthetic images)
NUMBER_OF_IMAGES = 100000

# shuffle
RANDOM_SEED = 12345

# the range of the number of characters on each background image: (a, b) means a <= N <= b
NUM_OF_CHARS_RANGE = (6, 10)

# the range of font sizes: (a, b) means a <= N <= b
FONT_SIZE_RANGE = (32, 64)

# character patch threshold: larger 'a' means more vivid character images (a: 0~255)
CHARS_THRESH = 127

# splitting train/validation/test set (unit: %)
TRAIN_SET = 98
VALID_SET = 1
TEST_SET = 1

# display a synthesized image per 'N' numbers of samples
DISPLAY_SAMPLE_PER_N = 1000
# ---------------------------------------------------------------

# if the target directory (for output) already exists, remove it
try:
    shutil.rmtree(TARGET_ROOT_DIR)
    shutil.rmtree(TARGET_IMG_DIR)
    shutil.rmtree(TARGET_ANNO_DIR)
except OSError:
    pass

# make a new target directory
os.mkdir(TARGET_ROOT_DIR)
os.mkdir(TARGET_IMG_DIR)
os.mkdir(TARGET_ANNO_DIR)


def findImages(root_dir,
               file_format='.jpg'):
    filenames = []

    for (path, dirs, files) in os.walk(root_dir):
        fullpath = os.path.join(root_dir, path)
        for file in files:
            # get extension format of a file
            ext = os.path.splitext(file)[-1]
            if ext == file_format:
                filenames.append(os.path.join(fullpath, file))

    print("Found '{}' '{}' image files from the path: '{}'\n"
          .format(len(filenames), file_format, root_dir))

    return filenames


def shuffleFileLists(filenames,
                     random_seed=RANDOM_SEED):
    shuffled_index = list(range(len(filenames)))
    random.seed(random_seed)
    random.shuffle(shuffled_index)
    shuffled_filenames = [filenames[i] for i in shuffled_index]

    return shuffled_filenames


def drawNonOverlappingImages(bg_file,
                             char_files,
                             num_of_chars_range=NUM_OF_CHARS_RANGE,
                             font_size_range=FONT_SIZE_RANGE):
    def isOverlapped(prev_img, curr_img):
        # Input: (x, y, width, height)
        prev_center = (prev_img[0] + prev_img[2] / 2, prev_img[1] + prev_img[3] / 2)
        curr_center = (curr_img[0] + curr_img[2] / 2, curr_img[1] + curr_img[3] / 2)

        condition = (abs(prev_center[0] - curr_center[0]) > (prev_img[2] + curr_img[2]) / 2) | \
                    (abs(prev_center[1] - curr_center[1]) > (prev_img[3] + curr_img[3]) / 2)

        return not condition

    def patchImage(bg_img, char_img, pos_x, pos_y, font_size):

        char_img = cv2.resize(char_img, (font_size, font_size))
        roi = bg_img[pos_y:pos_y + font_size, pos_x:pos_x + font_size]

        char_img_gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(char_img_gray, CHARS_THRESH, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)

        bg_img_out = cv2.bitwise_and(roi, roi, mask=mask_inv)
        char_img_roi = cv2.bitwise_and(char_img, char_img, mask=mask)

        dst = cv2.add(bg_img_out, char_img_roi)
        bg_img[pos_y:pos_y + font_size, pos_x:pos_x + font_size] = dst

        return bg_img

    # read a background image file
    bg_img = cv2.imread(bg_file)
    bg_height, bg_width, bg_chn = bg_img.shape

    prev_images = [(0, 0, 0, 0)]

    for file in char_files:

        found_new_pos = False

        while not found_new_pos:
            font_size = random.randint(font_size_range[0], font_size_range[1])

            x = random.randint(0, bg_width - font_size)
            y = random.randint(0, bg_height - font_size)

            is_overlapped = [isOverlapped(prev_img, (x, y, font_size, font_size)) for prev_img in prev_images]

            if not any(is_overlapped):
                found_new_pos = True
                prev_images.append((x, y, font_size, font_size))

                #                 bg_img = cv2.rectangle(bg_img, (x, y), (x+font_size, y+font_size), (0,255,0), 3)
                char_img = cv2.imread(file)
                bg_img = patchImage(bg_img, char_img, x, y, font_size)

    return bg_img, prev_images[1:]


def makeXML(file_path, bg_img, char_names, char_info=(0, 0, 0, 0)):
    # char_info: (x, y, font_size, font_size)

    def indent(elem, level=0):
        i = "\n" + level * "    "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "    "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    bg_height, bg_width, bg_chn = bg_img.shape

    annotation = Element("annotation")
    SubElement(annotation, "folder").text = TARGET_ROOT_DIR
    SubElement(annotation, "filename").text = str(file_path.split('/')[-1])
    size = Element("size")
    annotation.append(size)
    SubElement(size, "width").text = str(bg_width)
    SubElement(size, "height").text = str(bg_height)
    SubElement(size, "depth").text = '3'

    for i, char in enumerate(char_names):
        obj = Element("object")
        annotation.append(obj)
        SubElement(obj, "name").text = char
        bndbox = Element("bndbox")
        obj.append(bndbox)

        x, y, width, height = char_info[i]

        SubElement(bndbox, "xmin").text = str(x)
        SubElement(bndbox, "ymin").text = str(y)
        SubElement(bndbox, "xmax").text = str(x + width)
        SubElement(bndbox, "ymax").text = str(y + height)

    indent(annotation)
    #     dump(annotation)
    file_name = file_path.split('/')[-1].split('.')[0]
    ElementTree(annotation).write(os.path.join(TARGET_ANNO_DIR, file_name + '.xml'), encoding='utf-8')


# list up & shuffle character image files
char_filenames = findImages(root_dir=SOURCE_CHAR_DIR, file_format=SOURCE_CHAR_FORMAT)

shuffled_char_filenames = shuffleFileLists(char_filenames)

# list up background image files
bg_filenames = findImages(root_dir=SOURCE_BG_DIR, file_format=SOURCE_BG_FORMAT)

shuffled_bg_filenames = shuffleFileLists(bg_filenames)

# list up how to distribute char images on background images
char_distribute_plan = []

for _ in range(NUMBER_OF_IMAGES):
    random_number = random.randint(NUM_OF_CHARS_RANGE[0], NUM_OF_CHARS_RANGE[1])
    char_distribute_plan.append(random_number)

# patch character images on background images according to the plan
num_chars = len(shuffled_char_filenames)
num_bgs = len(shuffled_bg_filenames)

char_idx = 0
bg_idx = 0
for i, num_of_images in enumerate(char_distribute_plan):
    synth_img, char_info = drawNonOverlappingImages(
        shuffled_bg_filenames[bg_idx],
        shuffled_char_filenames[char_idx:char_idx + num_of_images])

    # save a synthetic image
    file_path = os.path.join(TARGET_IMG_DIR, str(i) + TARGET_FILE_FORMAT)
    cv2.imwrite(file_path, synth_img)

    # Save a xml file
    char_names = [filename.split('/')[-2]
                  for filename
                  in shuffled_char_filenames[char_idx:char_idx + num_of_images]]

    makeXML(file_path, synth_img, char_names, char_info)

    char_idx = (char_idx + num_of_images) % num_chars
    bg_idx = i % num_bgs

    if not (i % DISPLAY_SAMPLE_PER_N):
        print("Synthetic image output: # {}".format(i))
        plt.imshow(synth_img)
        plt.show()

# split train/val/test set
shuffled_index = list(range(NUMBER_OF_IMAGES))
random.seed(RANDOM_SEED)
random.shuffle(shuffled_index)

num_train = int(NUMBER_OF_IMAGES * TRAIN_SET / (TRAIN_SET + VALID_SET + TEST_SET))
num_valid = int(NUMBER_OF_IMAGES * VALID_SET / (TRAIN_SET + VALID_SET + TEST_SET))
num_test = NUMBER_OF_IMAGES - num_train - num_valid

with open(TARGET_ROOT_DIR + "/train.txt", "w") as wf:
    for index in shuffled_index[0:num_train]:
        wf.write(str(index) + '\n')

with open(TARGET_ROOT_DIR + "/val.txt", "w") as wf:
    for index in shuffled_index[num_train:num_train + num_valid]:
        wf.write(str(index) + '\n')

with open(TARGET_ROOT_DIR + "/trainval.txt", "w") as wf:
    for index in shuffled_index[0:num_train + num_valid]:
        wf.write(str(index) + '\n')

with open(TARGET_ROOT_DIR + "/test.txt", "w") as wf:
    for index in shuffled_index[num_train + num_valid:]:
        wf.write(str(index) + '\n')

with open(TARGET_ROOT_DIR + "/labels.txt", "w") as wf:
    for label in os.listdir(SOURCE_CHAR_DIR):
        wf.write(str(label) + '\n')

print("Train / Valid / Test : {} / {} / {}".format(num_train, num_valid, num_test))
print("Output path: {}".format(TARGET_ROOT_DIR))
print("Made {} synthetic images.".format(NUMBER_OF_IMAGES))