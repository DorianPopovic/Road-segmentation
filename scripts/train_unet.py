import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12,12)
from U_net import *
from losses import *
import os
import random
from datagen import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from losses import *
import zipfile
from generate_augmented_images import *



with zipfile.ZipFile('../data/training.zip', 'r') as zip_ref:
    zip_ref.extractall('../data')

## Seeding
seed = 2
random.seed = seed
np.random.seed = seed
tf.seed = seed
epochs = 100

dataset_path = "../data/"
train_path = os.path.join(dataset_path, "training/")


#We perform data augmentation:

imgs_dir='../data/training/images/'
gt_imgs_dir='../data/training/groundtruth/'

print('Augmenting data size to 1600 images')
generate_flipped_images(imgs_dir, gt_imgs_dir)
generate_transposed_images(imgs_dir, gt_imgs_dir)
generate_rotated_by_45_degrees_images(imgs_dir, gt_imgs_dir)
print('Data augmentation finished')


train_ids = []
path = train_path + 'images'

files = os.listdir(path)
for name in files:
    train_ids.append(name)

random.Random(seed).shuffle(train_ids)

image_size = 400
batch_size = 6

val_data_size = 400

valid_ids = train_ids[:val_data_size]
train_ids = train_ids[val_data_size:]

save_model_path = '/tmp/weights.h5'
cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_dice_loss', save_best_only=True, verbose=1)

train_gen = DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
valid_gen = DataGen(valid_ids, train_path, image_size=image_size, batch_size=batch_size)

train_steps = len(train_ids)//batch_size
valid_steps = len(valid_ids)//batch_size

model=ResUNet(image_size,kernel_size=(5,5))
model.compile('adam',loss=bce, metrics=[dice_loss, bce, tf.keras.metrics.Accuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

history=model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps,
                    epochs=epochs,callbacks= [cp])

dice = history.history['dice_loss']
val_dice = history.history['val_dice_loss']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, dice, label='Training Dice Loss')
plt.plot(epochs_range, val_dice, label='Validation Dice Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Dice Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()
plt.savefig('Loss.png')