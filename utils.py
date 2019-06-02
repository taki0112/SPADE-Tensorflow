import tensorflow as tf
from tensorflow.contrib import slim
from scipy import misc
import os, random
import numpy as np
from glob import glob
from tqdm import tqdm
from ast import literal_eval
import cv2

class Image_data:

    def __init__(self, img_height, img_width, channels, segmap_ch, dataset_path, augment_flag):
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.segmap_channel = segmap_ch
        self.augment_flag = augment_flag

        self.dataset_path = dataset_path
        self.img_dataset_path = os.path.join(dataset_path, 'image')
        self.segmap_dataset_path = os.path.join(dataset_path, 'segmap')
        self.segmap_test_dataset_path = os.path.join(dataset_path, 'segmap_test')

        self.image = []
        self.color_value_dict = {}
        self.segmap = []
        self.segmap_test = []

        self.set_x = set()


    def image_processing(self, filename, segmap):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels, dct_method='INTEGER_ACCURATE')
        img = tf.image.resize_images(x_decode, [self.img_height, self.img_width])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        segmap_x = tf.read_file(segmap)
        segmap_decode = tf.image.decode_jpeg(segmap_x, channels=self.segmap_channel, dct_method='INTEGER_ACCURATE')
        segmap_img = tf.image.resize_images(segmap_decode, [self.img_height, self.img_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if self.augment_flag :
            augment_height_size = self.img_height + (30 if self.img_height == 256 else int(self.img_height * 0.1))
            augment_width_size = self.img_width + (30 if self.img_width == 256 else int(self.img_width * 0.1))

            p = random.random()
            if p > 0.5:
                img, segmap_img = augmentation(img, segmap_img, augment_height_size, augment_width_size)


        label_map = convert_from_color_segmentation(self.color_value_dict, segmap_img, tensor_type=True)
        segmap_onehot = tf.one_hot(label_map, len(self.color_value_dict))

        return img, segmap_img, segmap_onehot

    def test_image_processing(self, segmap):
        segmap_x = tf.read_file(segmap)
        segmap_decode = tf.image.decode_jpeg(segmap_x, channels=self.segmap_channel, dct_method='INTEGER_ACCURATE')
        segmap_img = tf.image.resize_images(segmap_decode, [self.img_height, self.img_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        label_map = convert_from_color_segmentation(self.color_value_dict, segmap_img, tensor_type=True)
        segmap_onehot = tf.one_hot(label_map, len(self.color_value_dict))

        return segmap_img, segmap_onehot

    def preprocess(self):

        self.image = glob(self.img_dataset_path + '/*.*')
        self.segmap = glob(self.segmap_dataset_path + '/*.*')
        self.segmap_test = glob(self.segmap_test_dataset_path + '/*.*')
        segmap_label_path = os.path.join(self.dataset_path, 'segmap_label.txt')

        if os.path.exists(segmap_label_path) :
            print("segmap_label exists ! ")

            with open(segmap_label_path, 'r') as f:
                self.color_value_dict = literal_eval(f.read())

        else :
            print("segmap_label no exists ! ")
            x_img_list = []
            label = 0
            for img in tqdm(self.segmap) :

                if self.segmap_channel == 1 :
                    x = cv2.imread(img, flags=cv2.IMREAD_GRAYSCALE)
                else :
                    x = cv2.imread(img, flags=cv2.IMREAD_COLOR)
                    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

                x = cv2.resize(x, dsize=(self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)

                if self.segmap_channel == 1 :
                    x = np.expand_dims(x, axis=-1)

                h, w, c = x.shape

                x_img_list.append(x)

                for i in range(h) :
                    for j in range(w) :
                        if tuple(x[i, j, :]) not in self.color_value_dict.keys() :
                            self.color_value_dict[tuple(x[i, j, :])] = label
                            label += 1

            with open(segmap_label_path, 'w') as f :
                f.write(str(self.color_value_dict))

        print()

def load_segmap(dataset_path, image_path, img_width, img_height, img_channel):
    segmap_label_path = os.path.join(dataset_path, 'segmap_label.txt')

    with open(segmap_label_path, 'r') as f:
        color_value_dict = literal_eval(f.read())


    if img_channel == 1:
        segmap_img = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
    else :
        segmap_img = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        segmap_img = cv2.cvtColor(segmap_img, cv2.COLOR_BGR2RGB)


    segmap_img = cv2.resize(segmap_img, dsize=(img_width, img_height), interpolation=cv2.INTER_NEAREST)

    if img_channel == 1:
        segmap_img = np.expand_dims(segmap_img, axis=-1)

    label_map = convert_from_color_segmentation(color_value_dict, segmap_img, tensor_type=False)

    segmap_onehot = get_one_hot(label_map, len(color_value_dict))

    segmap_onehot = np.expand_dims(segmap_onehot, axis=0)

    """
    segmap_x = tf.read_file(image_path)
    segmap_decode = tf.image.decode_jpeg(segmap_x, channels=img_channel, dct_method='INTEGER_ACCURATE')
    segmap_img = tf.image.resize_images(segmap_decode, [img_height, img_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    label_map = convert_from_color_segmentation(color_value_dict, segmap_img, tensor_type=True)

    segmap_onehot = tf.one_hot(label_map, len(color_value_dict))

    segmap_onehot = tf.expand_dims(segmap_onehot, axis=0)
    """

    return segmap_onehot

def load_style_image(image_path, img_width, img_height, img_channel):

    if img_channel == 1 :
        img = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
    else :
        img = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, dsize=(img_width, img_height))

    if img_channel == 1 :
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
    else :
        img = np.expand_dims(img, axis=0)

    img = preprocessing(img)

    return img


def preprocessing(x):
    x = x/127.5 - 1 # -1 ~ 1
    return x

def augmentation(image, segmap, augment_height, augment_width):
    seed = random.randint(0, 2 ** 31 - 1)

    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize_images(image, [augment_height, augment_width])
    image = tf.random_crop(image, ori_image_shape, seed=seed)

    ori_segmap_shape = tf.shape(segmap)
    segmap = tf.image.random_flip_left_right(segmap, seed=seed)
    segmap = tf.image.resize_images(segmap, [augment_height, augment_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    segmap = tf.random_crop(segmap, ori_segmap_shape, seed=seed)

    return image, segmap

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return (images+1.) / 2


def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ('true')

def get_one_hot(targets, nb_classes):

    x = np.eye(nb_classes)[targets]

    return x

def convert_from_color_segmentation(color_value_dict, arr_3d, tensor_type=False):

    if tensor_type :
        arr_2d = tf.zeros(shape=[tf.shape(arr_3d)[0], tf.shape(arr_3d)[1]], dtype=tf.uint8)

        for c, i in color_value_dict.items() :
            color_array = tf.reshape(np.asarray(c, dtype=np.uint8), shape=[1, 1, -1])
            condition = tf.reduce_all(tf.equal(arr_3d, color_array), axis=-1)
            arr_2d = tf.where(condition, tf.cast(tf.fill(tf.shape(arr_2d), i), tf.uint8), arr_2d)

        return arr_2d

    else :
        arr_2d = np.zeros((np.shape(arr_3d)[0], np.shape(arr_3d)[1]), dtype=np.uint8)

        for c, i in color_value_dict.items():
            color_array = np.asarray(c, np.float32).reshape([1, 1, -1])
            m = np.all(arr_3d == color_array, axis=-1)
            arr_2d[m] = i

        return arr_2d

def pytorch_xavier_weight_factor(gain=0.02, uniform=False) :

    if uniform :
        factor = gain * gain
        mode = 'FAN_AVG'
    else :
        factor = (gain * gain) / 1.3
        mode = 'FAN_AVG'

    return factor, mode, uniform

def pytorch_kaiming_weight_factor(a=0.0, activation_function='relu', uniform=False) :

    if activation_function == 'relu' :
        gain = np.sqrt(2.0)
    elif activation_function == 'leaky_relu' :
        gain = np.sqrt(2.0 / (1 + a ** 2))
    elif activation_function == 'tanh' :
        gain = 5.0 / 3
    else :
        gain = 1.0

    if uniform :
        factor = gain * gain
        mode = 'FAN_IN'
    else :
        factor = (gain * gain) / 1.3
        mode = 'FAN_IN'

    return factor, mode, uniform