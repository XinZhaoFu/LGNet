from tensorflow.keras.metrics import MeanIoU
import numpy as np


def get_iou(label_value, predict_value, num_classes=2):
    """

    :param label_value:
    :param predict_value:
    :param num_classes:
    :return:
    """
    m = MeanIoU(num_classes=num_classes)
    _ = m.update_state(y_true=label_value, y_pred=predict_value)
    iou = m.result().numpy()
    return iou


def get_dice(label_value, predict_value):
    """

    :param label_value:
    :param predict_value:
    :return:
    """
    iou = get_iou(label_value, predict_value)
    dice = (2 * iou) / (1 + iou)
    return dice


def get_vcdr(predict):
    """

    :param predict:
    :return:
    """
    od_rows, od_cols = np.where(predict == 127)
    odr = max(od_rows) - min(od_rows)

    oc_rows, oc_cols = np.where(predict == 255)
    ocr = max(oc_rows) - min(oc_rows)

    return ocr / odr


