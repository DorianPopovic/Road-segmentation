from tensorflow import keras
import cv2
import numpy as np
import os

class DataGen(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=8, image_size=400):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()

    def __load__(self, id_name):
        ## Path
        image_path = os.path.join(self.path, "images", id_name)
        mask_path = os.path.join(self.path, "groundtruth", id_name)

        ## Reading Image
        image = cv2.imread(image_path)

        ##Reading Mask
        mask = cv2.imread(mask_path, 0)
        mask = np.expand_dims(mask, axis=-1)

        ## Normalizaing
        image = image / 255.0
        mask = mask / 255.0

        return image, mask

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index * self.batch_size

        files_batch = self.ids[index * self.batch_size: (index + 1) * self.batch_size]

        image = []
        mask = []

        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)

        image = np.array(image)
        mask = np.array(mask)

        return image, mask

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))