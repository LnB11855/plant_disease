from __future__ import print_function, division
from PIL import Image
import math
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Reshape
from keras.layers.convolutional import UpSampling2D
import numpy as np
import os
import imageio
import pickle
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from skimage.transform import resize as imresize
from keras.layers import BatchNormalization
from keras.optimizers import Adam, SGD
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from subprocess import check_output
from keras import backend as K
print(check_output(["ls", "D:/2ndPlant/crowdai_plantvillage"]).decode("utf8"))
class ACGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 38
        self.latent_dim = 100
        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses,
            optimizer=optimizer,
            metrics=['accuracy'])
        self.generator = self.build_generator()
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])
        self.discriminator.trainable = False
        valid, target_label = self.discriminator(img)
        self.combined = Model([noise, label], [valid, target_label])
        def custom_objective(y_true, y_pred):
            epi=0.22

            #out = -epi+(1-epi)*K.sparse_categorical_crossentropy(y_true, y_pred)
            out =-epi*(K.mean(K.log(y_pred+0.00000001)))+(1-epi)*K.sparse_categorical_crossentropy(y_true, y_pred)
            print(out)
            return out
        self.combined.compile(loss= ['binary_crossentropy',custom_objective],
            optimizer=optimizer)


    def build_generator(self):
        model = Sequential()
        model.add(Dense(128 * 16 * 16, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((16, 16, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))
        model.summary()
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, 100)(label))
        model_input = multiply([noise, label_embedding])
        img = model(model_input)
        return Model([noise, label], img)

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.summary()
        img = Input(shape=self.img_shape)
        features = model(img)
        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes+1, activation="softmax")(features)

        return Model(img, [validity, label])

    def train(self, epochs, batch_size=128, sample_interval=50):
        # X_train, x_valid, y_train, y_valid = reader();


        with open('D:\\2ndPlant\\regularization.pickle', 'rb') as f:
            X_train, y_train=pickle.load( f)
        X_train, x_valid, y_train, y_valid = train_test_split(
            X_train,
            y_train,
            shuffle=True,
            train_size=0.05,
            random_state=RANDOM_STATE
        )
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            noise = np.random.normal(0, 1, (batch_size, 100))
            sampled_labels = np.random.randint(0, self.num_classes, (batch_size, 1))
            gen_imgs = self.generator.predict([noise, sampled_labels])
            img_labels = y_train[idx]
            fake_labels = self.num_classes * np.ones(img_labels.shape)
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, fake_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])
            print ("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))
            if epoch % sample_interval == 0:
                self.save_model()
                self.sample_images(epoch)


    def sample_images(self, epoch):
        r, c = 10, self.num_classes
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.array([num for _ in range(r) for num in range(c)])
        gen_imgs = self.generator.predict([noise, sampled_labels])
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(6, 7)
        cnt = 0
        for i in range(6):
            for j in range(7):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0:3])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()

    def save_model(self):
        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])
        save(self.generator, "generator")
        save(self.discriminator, "discriminator")
def img_reshape(img):
    img = imresize(img, (64, 64, 3))
    return img
def img_label(path):
    return str(str(path.split('\\')[-1]))
def img_class(path):
    return str(path.split('\\')[-2])
def fill_dict(paths, some_dict):
    text = ''
    if 'train' in paths[0]:
        text = 'Start fill train_dict'
    elif 'test' in paths[0]:
        text = 'Start fill test_dict'

    for p in tqdm(paths, ascii=True, ncols=85, desc=text):
        img = imageio.imread(p)
        img = img_reshape(img)
        some_dict['image'].append(img)
        some_dict['label'].append(img_label(p))
        if 'train' in paths[0]:
            some_dict['class'].append(img_class(p))

    return some_dict
def reader():
    file_ext = []
    train_path = []
    test_path = []

    for root, dirs, files in os.walk('D:/2ndPlant/crowdai_plantvillage'):
        if dirs != []:
            print('Root:\n' + str(root))
            print('Dirs:\n' + str(dirs))
        else:
            for f in files:
                ext = os.path.splitext(str(f))[1][1:]

                if ext not in file_ext:
                    file_ext.append(ext)

                if 'train' in root:
                    path = os.path.join(root, f)
                    train_path.append(path)
                elif 'test' in root:
                    path = os.path.join(root, f)
                    test_path.append(path)
    train_dict = {
        'image': [],
        'label': [],
        'class': []
    }
    test_dict = {
        'image': [],
        'label': []
    }

    train_dict = fill_dict(train_path, train_dict)
    # test_dict = fill_dict(test_path, test_dict)

    X_train = np.array(train_dict['image'])
    y_train = np.array([CLASS[l] for l in train_dict['class']])
    X_train = (X_train.astype(np.float32) - 0.5) * 2
    y_train = y_train.reshape(-1, 1)
    with open('D:\\2ndPlant\\regularization.pickle', 'wb') as f:
        pickle.dump([X_train, y_train], f,protocol=pickle.HIGHEST_PROTOCOL)
    X_train, x_valid, y_train, y_valid = train_test_split(
        X_train,
        y_train,
        shuffle=True,
        train_size=0.05,
        random_state=RANDOM_STATE
    )
    return  X_train, x_valid, y_train, y_valid

if __name__ == '__main__':
    CLASS = {"c_" + str(i): i for i in range(38)}
    RANDOM_STATE = 11
    acgan = ACGAN()
    acgan.train(epochs=300000, batch_size=128, sample_interval=200)