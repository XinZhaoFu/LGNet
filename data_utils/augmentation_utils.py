from numpy.random import randint, uniform, shuffle
import cv2
import numpy as np
from tqdm import tqdm
from data_utils.utils import one_data_adjust, center_crop


def center_random_rotate_crop(img, label, center_offset=0):
    """
    先对图像进行随机旋转
    再以视杯区域中心点经小幅随机偏移后作为裁剪中心点 裁剪随机大小的图像

    :param img:
    :param label:
    :param center_offset:
    :return:
    """
    img, label = random_rotate(img, label, is_restrict=True)

    img_size, _, img_channel = img.shape
    label_size, _ = label.shape
    if label_size != img_size:
        img = cv2.resize(img, dsize=(label_size, label_size))
        img_size = label_size

    rows, cols = np.where(label == 255)
    if center_offset != 0:
        center_row = int(sum(rows) / len(rows)) + randint(-center_offset, center_offset)
        center_col = int(sum(cols) / len(cols)) + randint(-center_offset, center_offset)
    else:
        center_row = int(sum(rows) / len(rows))
        center_col = int(sum(cols) / len(cols))
    part_crop_size = randint(img_size//4, img_size//3)

    row_left, row_right = max(center_row - part_crop_size, 0), min(center_row + part_crop_size, img_size)
    col_top, col_down = max(center_col - part_crop_size, 0), min(center_col + part_crop_size, img_size)
    crop_img_rows, crop_img_cols = row_right - row_left, col_down - col_top

    crop_img = np.zeros(shape=(crop_img_rows, crop_img_cols, img_channel), dtype=np.uint8)
    crop_label = np.zeros(shape=(crop_img_rows, crop_img_cols), dtype=np.uint8)

    crop_img[:, :, :] = img[row_left:row_right, col_top:col_down, :]
    crop_label[:, :] = label[row_left:row_right, col_top:col_down]

    return crop_img, crop_label


def random_crop(img, label):
    """
    随机裁剪  因为img尺寸和label尺寸不一定对应 resize_rate是用于调整比例的

    :param img:
    :param label:
    :return:
    """
    img_rows, img_cols, img_channel = img.shape
    label_rows, label_cols = label.shape

    resize_rate_row = label_rows / img_rows
    resize_rate_col = label_cols / img_cols

    img_random_crop_row_length = randint(img_rows // 4, (img_rows // 4) * 3)
    img_random_crop_col_length = randint(img_cols // 4, (img_cols // 4) * 3)
    label_random_crop_row_length = int(img_random_crop_row_length * resize_rate_row)
    label_random_crop_col_length = int(img_random_crop_col_length * resize_rate_col)

    img_random_crop_row_init = randint(1, img_rows - img_random_crop_row_length)
    img_random_crop_col_init = randint(1, img_cols - img_random_crop_col_length)
    label_random_crop_row_init = int(img_random_crop_row_init * resize_rate_row)
    label_random_crop_col_init = int(img_random_crop_col_init * resize_rate_col)

    crop_img = np.zeros(shape=(img_random_crop_row_length, img_random_crop_col_length, img_channel), dtype=np.uint8)
    crop_label = np.zeros(shape=(label_random_crop_row_length, label_random_crop_col_length), dtype=np.uint8)

    crop_img[:, :, :] = img[img_random_crop_row_init:img_random_crop_row_init + img_random_crop_row_length,
                        img_random_crop_col_init:img_random_crop_col_init + img_random_crop_col_length, :]
    crop_label[:, :] = label[label_random_crop_row_init: label_random_crop_row_init + label_random_crop_row_length,
                       label_random_crop_col_init: label_random_crop_col_init + label_random_crop_col_length]

    return crop_img, crop_label


def random_color_scale(img, alpha_rate=0.3, base_beta=30):
    """
    做一个颜色扰动 img = img * alpha + beta

    :param img:
    :param alpha_rate:
    :param base_beta:
    :return:
    """
    img_rows, img_cols, img_channels = img.shape

    for index in range(img_channels):
        temp_img_split = np.zeros(shape=(img_rows, img_cols), dtype=np.uint8)
        temp_img_split[:, :] = img[:, :, index]
        temp_img_split_beta = np.random.randint(-base_beta, base_beta, size=(img_rows, img_cols))
        alpha = uniform(1 - alpha_rate, 1 + alpha_rate)
        temp_img_split = temp_img_split * alpha + temp_img_split_beta

        (temp_rows, temp_cols) = np.where(temp_img_split < 0)
        temp_img_split[temp_rows, temp_cols] = 0
        (temp_rows, temp_cols) = np.where(temp_img_split > 255)
        temp_img_split[temp_rows, temp_cols] = 255

        img[:, :, index] = temp_img_split

    return img


def random_color_shuffle(img):
    """

    :param img:
    :return:
    """
    random_channel = [0, 1, 2]
    shuffle(random_channel)
    img[:, :, 0] = img[:, :, random_channel[0]]
    img[:, :, 1] = img[:, :, random_channel[1]]
    img[:, :, 2] = img[:, :, random_channel[2]]

    return img


def random_flip(img, label):
    """
    做一个随机翻转

    :param img:
    :param label:
    :return:
    """
    random_num = randint(0, 1)
    img = cv2.flip(img, random_num)
    label = cv2.flip(label, random_num)

    return img, label


def random_rotate(img, label, is_restrict=True):
    """

    :param img:
    :param label:
    :param is_restrict:
    :return:
    """
    img_rows, img_cols, _ = img.shape
    label_rows, label_cols = label.shape
    if img_rows != img_cols:
        img = cv2.resize(img, dsize=(img_rows, img_rows))
        label = cv2.resize(label, dsize=(label_rows, label_rows))

    if is_restrict:
        random_angle = randint(1, 3) * 90
    else:
        random_angle = randint(1, 359)
    img_rotate = cv2.getRotationMatrix2D((img_rows * 0.5, img_rows * 0.5), random_angle, 1)
    label_rotate = cv2.getRotationMatrix2D((label_rows * 0.5, label_rows * 0.5), random_angle, 1)

    img = cv2.warpAffine(img, img_rotate, (img_rows, img_rows))
    label = cv2.warpAffine(label, label_rotate, (label_rows, label_rows))

    return img, label


def cutout(img, mask_rate=0.3):
    """
    对图片进行cutout 遮盖位置随机
    遮盖长度为空时用默认值
    过拟合增大，欠拟合缩小，自行调节
    添加遮盖前 对图像一圈进行0填充

    :param img:
    :param mask_rate:
    :return:cutout后的图像
    """
    img_rows, img_cols, _ = img.shape
    mask_rows = int(img_rows * mask_rate)
    mask_cols = int(img_cols * mask_rate)
    region_center_x, region_center_y = randint(0, img_rows), randint(0, img_cols)
    region_x0, region_x1 = max(0, region_center_x - mask_rows // 2), min(img_rows, region_center_x + mask_rows // 2)
    region_y0, region_y1 = max(0, region_center_y - mask_cols // 2), min(img_cols, region_center_y + mask_cols // 2)

    img[region_x0:region_x1, region_y0:region_y1, :] = 0

    return img


def gridMask(img, rate=0.1):
    """
    对图片进行gridmask
    每行每列各十个 以边均匀十等分 共计10*10个单元块  在每一个单元块内进行遮盖
    每一单元块其边长 = mask长度、offset偏差和留白
    长方形没做适配
    若干次的经验来看，过拟合增大rate，欠拟合缩小rate，但毕竟只是经验值

    :param img:
    :param rate: mask长度与一单元块边长的比值
    :return: gridmask后的图像
    """
    img_rows, img_cols, img_channel = img.shape
    fill_img_rows_length = int(img_rows + 0.2 * img_rows)
    fill_img_cols_length = int(img_cols + 0.2 * img_cols)
    rows_offset = randint(0, int(0.1 * fill_img_rows_length))
    cols_offset = randint(0, int(0.1 * fill_img_cols_length))
    rows_mask_length = int(0.1 * fill_img_rows_length * rate)
    cols_mask_length = int(0.1 * fill_img_cols_length * rate)

    fill_img = np.zeros((fill_img_rows_length, fill_img_cols_length, img_channel), dtype=np.uint8)
    fill_img[int(0.1 * img_rows):int(0.1 * img_rows) + img_rows, int(0.1 * img_cols):int(0.1 * img_cols) + img_cols] = img

    for width_num in range(10):
        for length_num in range(10):
            length_base_patch = int(0.1 * fill_img_rows_length * length_num) + rows_offset
            width_base_patch = int(0.1 * fill_img_cols_length * width_num) + cols_offset
            fill_img[length_base_patch:length_base_patch + rows_mask_length,
            width_base_patch:width_base_patch + cols_mask_length] = 0

    img = fill_img[int(0.1 * img_rows):int(0.1 * img_rows) + img_rows,
          int(0.1 * img_cols):int(0.1 * img_cols) + img_cols]

    return img


def random_filling(img, label):
    """
    目标是在图片周围填充一圈空值  实现方式是生成一个大的空值图 找一随机位置将原图填进去

    :param img:
    :param label:
    :return:
    """
    img = cv2.resize(img, dsize=(256, 256))
    label = cv2.resize(label, dsize=(256, 256), interpolation=cv2.INTER_AREA)
    # _, label = cv2.threshold(label, 0, 255, cv2.THRESH_BINARY)

    filling_img = np.zeros(shape=(512, 512, 3))
    filling_label = np.zeros(shape=(512, 512))

    random_x, random_y = randint(0, 256), randint(0, 256)

    filling_img[random_x:random_x + 256, random_y:random_y + 256] = img
    filling_label[random_x:random_x + 256, random_y:random_y + 256] = label

    return filling_img, filling_label


def augmentation_process(ori_img, ori_label, augmentation_rate, is_center_crop=True):
    """

    :param ori_img:
    :param ori_label:
    :param augmentation_rate: 扩增倍数
    :param is_center_crop:
    :return: aug_img_list, aug_label_list
    """
    aug_img_list, aug_label_list = [], []

    for index in range(augmentation_rate):
        choice_list = [randint(0, 2) for _ in range(5)]
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
            if is_center_crop:
                img, label = center_random_rotate_crop(img, label)
            else:
                img, label = random_crop(img, label)
        if choice_list[4]:
            if randint(0, 2) == 1:
                img = cutout(img)
            else:
                img = gridMask(img, rate=0.4)

        aug_img_list.append(img)
        aug_label_list.append(label)

    return aug_img_list, aug_label_list


def augmentation_distribution(img_file_list, label_file_list, target_img_path, target_label_path, augmentation_rate):
    """

    :param img_file_list:
    :param label_file_list:
    :param target_img_path:
    :param target_label_path:
    :param augmentation_rate:
    :return:
    """
    assert len(img_file_list) == len(label_file_list)
    for img_file, label_file in tqdm(zip(img_file_list, label_file_list), total=len(img_file_list)):
        ori_img = cv2.imread(img_file)
        ori_label = cv2.imread(label_file, 0)
        img_name = (img_file.split('/')[-1]).split('.')[0]
        label_name = (label_file.split('/')[-1]).split('.')[0]

        # aug_img_list, aug_label_list = augmentation_process(ori_img, ori_label, augmentation_rate, is_center_crop=True)
        #
        # img, label = one_data_adjust(ori_img, ori_label)
        # cv2.imwrite(target_img_path + img_name + '_' + '{:0>2d}'.format(0) + '.jpg', img)
        # cv2.imwrite(target_label_path + label_name + '_' + '{:0>2d}'.format(0) + '.png', label)

        # for index in range(augmentation_rate):
        #     aug_img, aug_label = one_data_adjust(aug_img_list[index], aug_label_list[index])
        #     cv2.imwrite(target_img_path + img_name + '_' + '{:0>2d}'.format(index+1) + '.jpg', aug_img)
        #     cv2.imwrite(target_label_path + label_name + '_' + '{:0>2d}'.format(index+1) + '.png', aug_label)

        crop_img, crop_label = center_crop(ori_img, ori_label)
        crop_aug_img_list, crop_aug_label_list = augmentation_process(crop_img, crop_label, augmentation_rate, is_center_crop=True)

        crop_img, crop_label = one_data_adjust(crop_img, crop_label)
        cv2.imwrite(target_img_path + img_name + '_crop_' + '{:0>2d}'.format(0) + '.jpg', crop_img)
        cv2.imwrite(target_label_path + label_name + '_crop_' + '{:0>2d}'.format(0) + '.png', crop_label)

        for index in range(augmentation_rate):
            # crop_aug_img, crop_aug_label = crop_aug_img_list[index], crop_aug_label_list[index]
            crop_aug_img, crop_aug_label = one_data_adjust(crop_aug_img_list[index], crop_aug_label_list[index])
            cv2.imwrite(target_img_path + img_name + '_crop_' + '{:0>2d}'.format(index + 1) + '.jpg', crop_aug_img)
            cv2.imwrite(target_label_path + label_name + '_crop_' + '{:0>2d}'.format(index + 1) + '.png', crop_aug_label)