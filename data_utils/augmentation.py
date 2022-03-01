from data_utils.augmentation_utils import *


def augmentation_process(ori_img, ori_label, num_aug, nor_size):
    aug_img_list, aug_label_list = [], []

    for index in range(num_aug):
        choice_list = [randint(0, 1) for _ in range(4)]
        img = np.empty(shape=ori_img.shape, dtype=np.uint8)
        label = np.empty(shape=ori_label.shape, dtype=np.uint8)
        img[:, :, :], label[:, :] = ori_img, ori_label

        if choice_list[0]:
            img, label = random_flip(img, label)
        if choice_list[1]:
            img, label = random_rotate(img, label)
        if choice_list[2]:
            img = random_color_shuffle(img)
            img = random_color_scale(img, alpha_rate=0.2)
        if choice_list[3]:
            img, label = center_random_rotate_crop(img, label)
        if randint(0, 1) == 1:
            img = cutout(img)
        else:
            img = gridMask(img, rate=0.2)

        aug_img = cv2.resize(img, dsize=(nor_size, nor_size))
        aug_label = cv2.resize(label, dsize=(nor_size, nor_size))

        aug_img_list.append(aug_img)
        aug_label_list.append(aug_label)

    return aug_img_list, aug_label_list

