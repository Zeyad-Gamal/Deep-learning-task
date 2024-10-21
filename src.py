

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import os
import cv2
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

import os
for dirname, _, filenames in os.walk('/content/drive/MyDrive/MKdata'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import os
import numpy as np
from PIL import Image

def load_images(root_folder, folder_names, target_size=(224,224)):
    X = []
    y = []
    counter = 0

    for folder_name in folder_names:
        folder_path = os.path.join(root_folder, folder_name)

        if os.path.isdir(folder_path):
            for filename in sorted(os.listdir(folder_path)):
                img_path = os.path.join(folder_path, filename)

                with Image.open(img_path) as img:
                    img_resized = img.resize(target_size)

                    img_array = np.array(img_resized)

                    X.append(img_array)

                    y.append(counter)

            counter += 1

    return np.array(X), np.array(y)

root_folder = '/content/drive/MyDrive/MKdata/D1/train'
selected_folders = ['Class (1)', 'Class (2)']
X_train, y_train = load_images(root_folder, selected_folders)
root_folder = '/content/drive/MyDrive/MKdata/D1/test'
X_test, y_test = load_images(root_folder, selected_folders)

print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)



y_train

y_train = np.array(y_train)
X_train = np.array(X_train)
y_test = np.array(y_test)
X_test = np.array(X_test)
print("X_train Shape: ", X_train.shape)
print("X_test Shape: ", X_test.shape)
print("y_train Shape: ", y_train.shape)
print("y_test Shape: ", y_test.shape)

X_train=X_train/255.0
X_test=X_test/255.0

batch_size = 32
img_size=224
epochs=5
NUM_CLASSES=2

d1_train_ds=ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
).flow_from_directory(
'/content/drive/MyDrive/MKdata/D1/train',
target_size = (img_size, img_size),
batch_size = batch_size,
class_mode = "categorical")
d1_val_ds=ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
).flow_from_directory(
'/content/drive/MyDrive/MKdata/D1/test',
target_size = (img_size, img_size),
batch_size = batch_size,
class_mode = "categorical")

from google.colab import drive
drive.mount('/content/drive')

import matplotlib.pyplot as plt


def plot_hist(hist):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].plot(hist.history["accuracy"])
    axs[0].plot(hist.history["val_accuracy"])
    axs[0].set_title("model accuracy")
    axs[0].set_ylabel("accuracy")
    axs[0].set_xlabel("epoch")
    axs[0].legend(["train", "validation"], loc="upper left")

    axs[1].plot(hist.history["loss"])
    axs[1].plot(hist.history["val_loss"])
    axs[1].set_title("model loss")
    axs[1].set_ylabel("loss")
    axs[1].set_xlabel("epoch")
    axs[1].legend(["train", "validation"], loc="upper left")

    plt.show()

def get_label_array(dsi):
    y=[]
    for i in range(len(dsi)):
        a,b=dsi[i]
        b=np.argmax(b,axis=1)
        y.append(b)

    y= np.concatenate(y)
    return y

y_test1=get_label_array(d1_val_ds)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def print_stats(y_test,y_pred) :
    confusion = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix\n')
    print(confusion)
    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

    print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
    print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

    print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

    print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))




import tensorflow as tf
from tensorflow.keras.metrics import Precision, SpecificityAtSensitivity, SensitivityAtSpecificity
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)

        self.true_positives = self.add_weight('true_positives', initializer='zeros')
        self.false_positives = self.add_weight('false_positives', initializer='zeros')
        self.false_negatives = self.add_weight('false_negatives', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(tf.math.round(y_pred), tf.float32)

        true_positives = tf.reduce_sum(y_true * y_pred)
        false_positives = tf.reduce_sum(tf.clip_by_value(y_pred - y_true, 0, 1))
        false_negatives = tf.reduce_sum(tf.clip_by_value(y_true - y_pred, 0, 1))

        self.true_positives.assign_add(true_positives)
        self.false_positives.assign_add(false_positives)
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())

        f1_score = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        return f1_score



def create_model_train_print_hist(model,modelname) :
    inputs = layers.Input(shape=(img_size,img_size, 3))
    output = model(inputs)
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(output)
    dropout_rate = 0.2
    dropout_layer = Dropout(dropout_rate)(global_average_layer)
    prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))(dropout_layer)
    model = tf.keras.Model(inputs=inputs, outputs=prediction_layer)


    model = tf.keras.Sequential( model)

    optimizer = tf.keras.optimizers.Adam(lr=0.001, weight_decay=0.001)

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy",tf.keras.metrics.Precision(),tf.keras.metrics.SpecificityAtSensitivity(0.5),tf.keras.metrics.SensitivityAtSpecificity(0.5),F1Score()])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    hist = model.fit(d1_train_ds, epochs=40,batch_size=32, validation_data=d1_val_ds, verbose=2, callbacks=[early_stopping])
    plot_hist(hist)
    model.save(modelname+'d1.h5')

    y_test_pred=model.predict(d1_val_ds)
    y_test_pred=np.argmax(y_test_pred,axis=1)
    print_stats(y_test1,y_test_pred)

model=EfficientNetB0(include_top=False, weights=None, classes=2)
create_model_train_print_hist(model,'mobilenetv2')




model.save('/content/drive/MyDrive/MKdata/D1/model_efficientnet.h5')

 loaded_model = tf.keras.models.load_model('/content/drive/MyDrive/MKdata/D1/model_efficientnet.h5')
model.compile( optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy",tf.keras.metrics.Precision(),tf.keras.metrics.SpecificityAtSensitivity(0.5),tf.keras.metrics.SensitivityAtSpecificity(0.5),F1Score()])

test_loss, test_accuracy = loaded_model.evaluate(d1_val_ds)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

from tensorflow.keras.models import load_model

# Load the saved model
saved_model_path = '/content/drive/MyDrive/MKdata/D1/model_efficientnet.h5'
loaded_model = load_model(saved_model_path)
test_loss, test_accuracy = loaded_model.evaluate(d1_val_ds)



