from __future__ import print_function, division
import sklearn as sk
from keras import backend as K
from sklearn.metrics import classification_report
from keras.layers import multiply
from keras.layers import Embedding
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Reshape
from keras.layers.convolutional import UpSampling2D
from subprocess import check_output
import numpy as np
import os
import imageio
import pickle
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import ZeroPadding2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from skimage.transform import resize as imresize
from tqdm import tqdm
import pandas as pd
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
        self.discriminator.load_weights(filepath='D:/2ndPlant/history0721/saved_model/discriminator_weights.hdf5')
        self.discriminator.compile(loss=losses,
            optimizer=optimizer,
            metrics=['accuracy'])
        self.generator = self.build_generator()
        self.generator.load_weights(filepath='D:/2ndPlant/history0721/saved_model/generator_weights.hdf5')
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])
        self.discriminator.trainable = False
        valid, target_label = self.discriminator(img)
        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss=losses,
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
        X_train, x_valid, y_train, y_valid = reader();
        y_train = y_train.reshape(-1, 1)
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
BATCH_SIZE = 100
# EPOCHS = 300
RANDOM_STATE = 11
MAX=0
ep=0
CLASS = {"c_"+str(i):i for i in range(38)}


INV_CLASS = {i:"c_"+str(i) for i in range(38)}



# Dense layers set
def dense_set(inp_layer, n, activation, drop_rate=0.):
    dp = Dropout(drop_rate)(inp_layer)
    dns = Dense(n)(dp)
    bn = BatchNormalization(axis=-1)(dns)
    act = Activation(activation=activation)(bn)
    return act


# Conv. layers set
def conv_layer(feature_batch, feature_map, kernel_size=(3, 3), strides=(1, 1), zp_flag=False):
    if zp_flag:
        zp = ZeroPadding2D((1, 1))(feature_batch)
    else:
        zp = feature_batch
    conv = Conv2D(filters=feature_map, kernel_size=kernel_size, strides=strides)(zp)
    bn = BatchNormalization(axis=3)(conv)
    act = LeakyReLU(1 / 10)(bn)
    return act

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
# simple model
def get_model():
    inp_img = Input(shape=(64, 64, 3))

    # 51
    conv1 = conv_layer(inp_img, 64, zp_flag=False)
    conv2 = conv_layer(conv1, 64, zp_flag=False)
    mp1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv2)
    # 23
    conv3 = conv_layer(mp1, 64, zp_flag=False)
    conv4 = conv_layer(conv3, 64, zp_flag=False)
    mp2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv4)
    # 9
    conv7 = conv_layer(mp2, 128, zp_flag=False)
    conv8 = conv_layer(conv7, 128, zp_flag=False)
    conv9 = conv_layer(conv8, 128, zp_flag=False)
    mp3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv9)
    # 1
    # dense layers
    flt = Flatten()(mp3)
    ds1 = dense_set(flt, 128, activation='tanh')
    out = dense_set(ds1, 38, activation='softmax')

    model = Model(inputs=inp_img, outputs=out)

    # The first 50 epochs are used by Adam opt.
    # Then 30 epochs are used by SGD opt.

    # mypotim = Adam(lr=2 * 1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    mypotim = SGD(lr=1 * 1e-1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=mypotim,
                  metrics=['accuracy',f1_m,precision_m, recall_m])
    model.summary()
    return model


def get_callbacks(filepath, patience=5):
    lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=1e-5, patience=patience, verbose=1)
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [lr_reduce, msave]


def train_model(img, target,a,b,c,d):
    #callbacks = get_callbacks(filepath='D:/2ndPlant/kaggle_seedling_classification/model_weight_SGD.hdf5', patience=6)

    gmodel = get_model()
    if d!=0:
     gmodel.load_weights(filepath='D:/2ndPlant/reg_new_model_weights.h5')

    gen = ImageDataGenerator(
        rotation_range=360.,
        width_shift_range=0.3,
        height_shift_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True
    )
    history = gmodel.fit_generator(gen.flow(img, target, batch_size=BATCH_SIZE),
                        steps_per_epoch=len(img) / BATCH_SIZE,
                         epochs=c,
                         verbose=1,
                         shuffle=True,
                         validation_data=(a, to_categorical(b)))
    # history = gmodel.fit(img, target, batch_size=BATCH_SIZE,
    #                      epochs=c, verbose=1, shuffle=True,
    #                      validation_data=(a, to_categorical(b)))
    model_json = gmodel.to_json()
    with open("reg_new_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    gmodel.save_weights("reg_new_model_weights.h5")
    # list all data in history
    print(history.history.keys())

    global MAX,ep,result_max
    if np.max(history.history['val_acc'])>MAX:
        MAX=np.max(history.history['val_acc'])
        y_pred_max = (gmodel.predict(a)).argmax(axis=-1)
        result_max = sk.metrics.confusion_matrix(b,y_pred_max)
        ep=d


    print((MAX,ep))
    dataframe = pd.DataFrame({'acc': history.history['acc'], 'val_acc': history.history['val_acc']})
    filename="record"+str(d)+".csv"
    dataframe.to_csv( filename, index=False, sep=',')
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Prediction Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()



# Resize all image to 51x51
def img_reshape(img):
    img = imresize(img, (64, 64, 3))
    return img


# get image tag
def img_label(path):
    return str(str(path.split('\\')[-1]))


# get plant class on image
def img_class(path):
    return str(path.split('\\')[-2])


# fill train and test dict
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


# read image from dir. and fill train and test dict
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
    return train_dict, test_dict



# train_dict, test_dict = reader()
# X_train = np.array(train_dict['image'])
# y_train = np.array([CLASS[l] for l in train_dict['class']])
# with open('D:\\2ndPlant\\regularization_test.pickle', 'wb') as f:
#     pickle.dump([X_train, y_train], f,protocol=pickle.HIGHEST_PROTOCOL)
with open('D:\\2ndPlant\\regularization_test.pickle', 'rb') as f:
    X_train, y_train=pickle.load(f)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train,
    y_train,
    shuffle=True,
    train_size=0.019,
    test_size=0.1,
    random_state=RANDOM_STATE
)
print("original train size",X_train.shape)
acgan = ACGAN()
r, c = 300, 38



# train_model(X_train, to_categorical(y_train),X_valid,y_valid,700,0)
# g_final_model = get_model()
# g_final_model.load_weights(filepath='D:/2ndPlant/reg_new_model_weights.h5')
# y_pred=(g_final_model.predict(X_valid)).argmax(axis=-1)
# result=sk.metrics.confusion_matrix(y_valid,y_pred)
# pd.DataFrame(result_max).to_csv("result_max_pure.csv")
# pd.DataFrame(result).to_csv("result_pure.csv")


for i in range(14):

    noise = np.random.normal(0, 1, (r * c, 100))
    sampled_labels = np.array([num for _ in range(r) for num in range(c)])
    gen_imgs = acgan.generator.predict([noise, sampled_labels])
    gen_validity, gen_plabels=acgan.discriminator.predict(gen_imgs)
    gen_imgs=gen_imgs[gen_validity[:,0]>0.6]
    gen_labels=np.asarray(np.argmax(gen_plabels, axis=1))
    gen_labels=gen_labels[gen_validity[:,0]>0.6]
    num_38 =np.sum(gen_labels==38)
    num_total,=gen_labels.shape
    print("success rate",(1-num_38/num_total))
    gen_imgs=gen_imgs[gen_labels<38]
    gen_labels = gen_labels[gen_labels<38]

    X_train_reg=np.concatenate((X_train,gen_imgs),axis=0)
    y_train_reg=np.concatenate((y_train,gen_labels),axis=0)
    print(X_train_reg.shape)
    train_model(X_train_reg, to_categorical(y_train_reg),X_valid,y_valid,50,i)

    #test_model(X_test, label)

g_final_model = get_model()
g_final_model.load_weights(filepath='D:/2ndPlant/reg_new_model_weights.h5')
y_pred=(g_final_model.predict(X_valid)).argmax(axis=-1)
result=sk.metrics.confusion_matrix(y_valid,y_pred)
pd.DataFrame(result_max).to_csv("result_max.csv")
pd.DataFrame(result).to_csv("result.csv")
