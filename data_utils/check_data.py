from glob import glob
import numpy as np
import cv2
from tqdm import tqdm


label_file_list = glob('../datasets/refuge_datasets/train/aug_label/' + '*.png')

for label_file in tqdm(label_file_list):
    label = cv2.imread(label_file, 0)

    point_sum = 0
    (rows, cols) = np.where(label == 0)
    point_sum += len(rows)
    (rows, cols) = np.where(label == 1)
    point_sum += len(rows)
    (rows, cols) = np.where(label == 2)
    point_sum += len(rows)

    if point_sum != 512*512:
        print(point_sum)
        break