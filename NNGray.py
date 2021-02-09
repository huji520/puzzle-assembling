import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Add, Activation, LeakyReLU, Conv2D, GlobalMaxPooling2D, Conv2DTranspose
from tensorflow.keras import Model
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import math
from scipy import ndimage
import random
from PIL import Image, ImageDraw
import copy
import click
import quantize


#### Constants ####
N = 32
MIN_SHAPE_PIXELS = 3
NUMBER_OF_SLICE = 4
RECT = 1
CIRCLE = 2
TRIANGLE = 3
###################


def create_shape(lower_radius, upper_radius):
    """
    creates an image of a random shape with random size (in a specific range)
    :param lower_radius: the lower bound of the shape radius
    :param upper_radius: the upper bound of the shape radius
    :return: binary image of shape (N,N), as np.array
    """
    r = random.randint(lower_radius, upper_radius)
    shape = random.choice([RECT, CIRCLE, TRIANGLE])
    im = Image.new('1', size=(N, N), color=0)
    draw = ImageDraw.Draw(im)
    x = (N - r) // 2
    y = x
    if shape == RECT:
        draw.rectangle(((y, x), (y + r, x + r)), fill=1)
    elif shape == CIRCLE:
        draw.ellipse(((y, x), (y + r, x + r)), fill=1)
    else:
        draw.polygon(((y, x), (y, x + r), (y + r, x + r // 2)), fill=1)

    return np.array(im, dtype=np.int)


def random_black_pixel(image):
    """
    return random black spot in image
    :param image:
    :return:
    """
    x = None
    y = None
    pixel = 0
    while pixel == 0:
        x = np.random.randint(0, image.shape[0])
        y = np.random.randint(0, image.shape[1])
        pixel = image[x, y]

    return x, y


class Line:

    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def is_larger(self, x, y):
        """
        detects if dot is above the line
        :param x: x index of 2D point
        :param y: y index of 2D point
        :return: boolean
        """
        vec1 = (self.p2[0] - self.p1[0], self.p2[1] - self.p1[1])
        vec2 = (- x + self.p2[0], -y + self.p2[1])
        cross = vec1[0]*vec2[1] - vec1[1]*vec2[0]
        return cross > 0


def random_frame_pixel(n):
    """
    random pixel on image frame
    :param n: image size
    :return: (x,y) pixel
    """
    x_or_y = np.random.randint(0, 2)
    max_or_min = np.random.randint(0, 2)
    offset = np.random.randint(0, n)

    out_pix = np.zeros(2)
    out_pix[x_or_y] = max_or_min * (n-1)
    out_pix[1 - x_or_y] = offset

    return out_pix


def slice_2image(image, m):
    """
    slicing a given image to 2 images
    :param image: the original image to slice (as np.array with shape (N,N))
    :param m: number of images needed to extracted at final step from each image
    :return: 2 slices of the original image. each slice is an image of shape (N,N), where the black part of
    the image is centralized. each slice must have at least 1 black pixel.
    """
    im_out1 = np.zeros(image.shape)
    im_out2 = np.zeros(image.shape)

    while np.sum(im_out1) < MIN_SHAPE_PIXELS * (m ** 1.5) or np.sum(im_out2) < MIN_SHAPE_PIXELS * (m ** 1.5):

        black_pixel = random_black_pixel(image)
        frame_pixel = random_frame_pixel(image.shape[0])
        line = Line(black_pixel, frame_pixel)

        x = np.arange(image.shape[0])
        y = np.arange(image.shape[1])
        x, y = np.meshgrid(x, y)

        bool_map = line.is_larger(x, y)

        im_out1 = image & bool_map
        im_out2 = image & ~bool_map

    return np.array([im_out1, im_out2])


def slice_image(image, m):
    """
    slicing a given image to m pieces
    :param image: the original image to slice (as np.array with shape (N,N))
    :param m: number of output pieces (must be a power of 2)
    :return: m slices of the original image. each slice is an image of shape (N,N), where the black part of
    the image is centralized. each slice must have at least 1 black pixel.
    """
    if m == 1:
        return image

    if not math.log(m, 2).is_integer() or not image.shape[0] == image.shape[1]:
        return None

    images = slice_2image(image, m // 2)

    return np.concatenate([slice_image(images[0], m // 2), slice_image(images[1], m // 2)]).reshape([m, image.shape[0], image.shape[1]])


def slice_image_color(image, m):
    """
    slicing a given image to m pieces with different colors
    :param image: the original image to slice (as np.array with shape (N,N))
    :param m: number of output pieces (must be a power of 2)
    :return: m slices of the original image. each slice is an image of shape (N,N), where the black part of
    the image is centralized. each slice must have at least 1 black pixel.
    """
    images = slice_image(image, m).astype(np.float)
    for i in range(1, m+1):
        color = i / m
        images[i-1] *= color
    return images


def reconstruct_images(parts):
    """
    reconstructs parts to an image
    :param parts: np array
    :return: image
    """
    out_image = copy.deepcopy(parts[0])
    for i in range(1, len(parts)):
        out_image += parts[i]
    return out_image


def generate_data_set(n, m):
    """
    creates a data set of sliced images and their labels (the original image)
    :param n: the number of pairs to generate
    :param m: number of slices for each image
    :return:
    """
    samples, labels = [], []
    for _ in range(n):
        label = create_shape()
        sample = slice_image(label, m)
        labels.append(label)
        samples.append(sample)
    pickle.dump(labels, open(f"labels_{n}_pairs_{m}_slices.pickle", 'w'))
    pickle.dump(samples, open(f"samples_{n}_pairs_{m}_slices.pickle", 'w'))
    return samples, labels


def transform_data(samples, labels, limit):
    """
    rotating the slices of a given sliced images, each slice in a random angle
    :param samples: the sliced images
    :param labels: the original images
    :param limit: how many images to rotate
    :return: the rotated samples and labels
    """
    for i in range(limit):
        result_batch = translate_and_rotate(samples[i])
        samples[i] = result_batch

    return samples, labels


def translate_single_data(img, center_of_mas):
    """
    Finding the center of mass and translating the image accordingly.
    x - vertical (axis 0), y- horizontal(axis 1)
    :param data:
    :return:
    """

    center_of_img_v = (img.shape[0] - 1) // 2
    center_of_img_h = (img.shape[1] - 1) // 2
    trans_vertical = int(center_of_img_v - center_of_mas[0])

    trans_horizontal = int(center_of_img_h - center_of_mas[1])
    img = np.roll(img, trans_vertical, axis=0)
    img = np.roll(img, trans_horizontal, axis=1)
    return img


def rotate_single_data(img):
    angle = np.random.choice(np.arange(0, 360, 10))
    rotated_im = ndimage.rotate(img, angle, reshape=False)
    return rotated_im


def find_center_of_mass(img):
    center_of_mas = ndimage.measurements.center_of_mass(img)
    return center_of_mas


def translate_and_rotate(img_batch):
    for i in range(len(img_batch)):
        center_of_mass = find_center_of_mass(img_batch[i])
        img_batch[i] = translate_single_data(img_batch[i], center_of_mass)
        img_batch[i] = rotate_single_data(img_batch[i])
    return img_batch


def get_data_set():
    TEST_SIZE = 1000
    samples = []
    labels = []
    for i in range(TEST_SIZE):
        if i % 1000 == 0:
            print("idx =", i)

        orig_image = create_shape(20, 30)
        sample = slice_image_color(orig_image, NUMBER_OF_SLICE)
        samples.append(sample)
        labels.append(reconstruct_images(sample))
    print("Done")

    labels = np.array(labels)
    samples = np.array(samples)
    # print("labels.shape =", labels.shape)
    # print("samples.shape =", samples.shape)
    return samples, labels



def train_generator():
    BATCH_SIZE = 32
    for j in range(50):
        samples = []
        labels = []
        for i in range(BATCH_SIZE):
            orig_image = create_shape(20, 30)
            sample = slice_image_color(orig_image, NUMBER_OF_SLICE)
            samples.append(sample)
            labels.append(reconstruct_images(sample))
        labels = np.array(labels)
        samples = np.array(samples)
        # print("labels.shape =", labels.shape)
        # print("samples.shape =", samples.shape)
        yield transform_data(samples, labels, BATCH_SIZE)

figure = 1
cutoff = 2000

x_test, y_test = get_data_set()
x_test, y_test = transform_data(x_test, y_test, 1000)


test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)



class Decoder(Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.reshape = Reshape((32, 32, 4))
        self.conv1 = Conv2D(16, 3, activation='relu', strides=2, padding='same')
        self.conv2 = Conv2D(32, 3, activation='relu', strides=2, padding='same')
        self.conv3 = Conv2D(64, 3, activation='relu', strides=2, padding='same')
        self.flatten = Flatten()
        self.d1 = Dense(512, activation='relu')
        self.d2 = Dense(128, activation='relu')

    def call(self, x):
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x

class Encoder(Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.d1 = Dense(512, activation='relu')
        self.d2 = Dense(2048, activation='relu')
        self.reshape = Reshape((8, 8, 32))
        self.deconv1 = Conv2DTranspose(16, 2, activation='relu', strides=2)
        self.deconv2 = Conv2DTranspose(1, 2, activation='relu', strides=2)

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return x

class AutoEncoder(Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, x):
        return self.encoder(self.decoder(x))


optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')


@tf.function
def train_step(model, images, labels, loss_object):
    with tf.GradientTape() as tape:
        predictions = model(images)
        predictions = tf.squeeze(predictions)
        loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)


def test_step(model, images, labels, loss_object):
    predictions = model(images)
    predictions = tf.squeeze(predictions)
    loss = loss_object(labels, predictions)
    test_loss(loss)


def train_model(model, epochs, loss_object, graph_train, graph_test):
    
    for epoch in range(epochs):
        for images, labels in train_generator():
            # print("labels.shape =", labels.shape)
            # print("images.shape =", images.shape)
            train_step(model, images, labels, loss_object)

        for images_test, labels_test in test_ds:
            test_step(model, images_test, labels_test, loss_object)

        if epoch % 20 == 0 and epoch > 0:
            model.save_weights(f"results/weights/weights_{epoch}.tf")
            indices = np.random.randint(0, 1000, 10)
            for j in indices:
                sample_test = x_test[j]
                label_test = y_test[j]
                reconstruction = model(sample_test[tf.newaxis, :, :]).numpy().squeeze()
                # try:
                #     reconstruction = quantize(reconstruction, 5, 20)[0]
                # except:
                    # continue
                global figure
                plt.figure(figure)
                figure += 1
                plt.subplot(2, 3, 1)
                plt.imshow(sample_test[0], cmap='gray')
                plt.subplot(2, 3, 2)
                plt.imshow(sample_test[1], cmap='gray')
                plt.subplot(2, 3, 3)
                plt.imshow(sample_test[2], cmap='gray')
                plt.subplot(2, 3, 4)
                plt.imshow(sample_test[3], cmap='gray')
                plt.subplot(2, 3, 5)
                plt.imshow(label_test, cmap='gray')
                plt.subplot(2, 3, 6)
                plt.imshow(reconstruction, cmap='gray')
                plt.savefig(f"test_show_{epoch}_{j}.png")

                plt.figure(figure)
                figure += 1
                plt.plot(graph_train, label="train")
                plt.plot(graph_test, label="test")
                plt.title("loss vs epoch")
                plt.xlabel("epoch")
                plt.ylabel("loss")

                if not os.path.isdir("results"):
                    os.mkdir("results")
                plt.savefig(f'results/loss_{epoch}.png')

        print(f'Epoch {epoch + 1}, Train loss: {train_loss.result()}, Test loss: {test_loss.result()}\n -- ')

        graph_train.append(train_loss.result())
        graph_test.append(test_loss.result())
        train_loss.reset_states()
        test_loss.reset_states()



@click.command()
@click.option('-e', '--epoch', default=200, help='Number of epochs')
@click.option('-l', '--load', default=0, help='Type the number of epoch of the weights that you want to load')
def run_gray_scale(epoch, load):
    print(f"Number of epochs: {epoch}")
    model = AutoEncoder()
    if load > 0:
        model.load_weights(f"results/weights/weights_{load}.tf")

    graph_train = []
    graph_test = []
    train_model(model, epoch, tf.keras.losses.MSE, graph_train, graph_test)
    model.summary()
    global figure
    plt.figure(figure)
    plt.plot(graph_train, label="train")
    plt.plot(graph_test, label="test")
    plt.title("loss vs epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")

    if not os.path.isdir("results"):
        os.mkdir("results")
    plt.savefig(f'results/{epoch}.png')
    
if __name__ == "__main__":
    run_gray_scale()
