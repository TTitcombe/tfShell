import numpy as np
import os
import cv2

class ImageLoader():
    '''A class to load batches of images as numpy arrays'''

    def __init__(self, path='./', batchSize=1):
        '''
        Input:
            path | string, path to folder containing images
        Determine image size, number of images
        '''
        filenames = os.listdir(path)
        self.N_files = len(filenames)
        self.files = filenames
        assert self.N_files > batchSize, "Batch size cannot be larger than total number of files"

        self.batchSize = batchSize
        self.steps = self.N_files // batchSize
        self.path = path
        self.current_n = 0

        if filenames != []:
            im_1 = cv2.imread(path + filenames[0])
            self.im_shape = im_1.size
        else:
            raise RuntimeError("No files present!")

    def next_batch(self):
        final_n = self.current_n + self.batchSize

        if final_n > (self.N_files - 1):
            final_n = self.N_files - 1

        images = np.zeros((final_n - self.current_n, self.im_shape))

        for i, file in enumerate(self.files):
            if i > self.current_n and i < final_n:
                im = cv2.imread(self.path + file)
                if im is not None:
                    im = im.flatten()
                    images[i-self.current_n] = im
                else:
                    print("Error loading {}".format(file))

        return images
