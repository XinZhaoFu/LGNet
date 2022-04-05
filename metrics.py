import cv2
from data_utils.metrics_utils import get_vcdr, get_iou, get_dice
from data_utils.utils import get_specific_type_file_list, file_consistency_check, get_oc_od


def get_metrics(predict_img_file_path, test_label_file_path):
    """

    :param predict_img_file_path:
    :param test_label_file_path:
    :return:
    """
    predict_img_file_list = get_specific_type_file_list(predict_img_file_path, 'png')
    test_label_file_list = get_specific_type_file_list(test_label_file_path, 'png')
    if not file_consistency_check(predict_img_file_list, test_label_file_list):
        predict_img_file_list.sort()
        test_label_file_list.sort()

    sum_mae, sum_od_iou, sum_oc_iou, sum_od_dice, sum_oc_dice = 0, 0, 0, 0, 0
    for predict_img_file, test_label_file in zip(predict_img_file_list, test_label_file_list):
        predict_img = cv2.imread(predict_img_file, 0)
        test_label = cv2.imread(test_label_file, 0)
        img_name = predict_img_file.split('/')[-1]

        predict_vcdr = get_vcdr(predict_img)
        test_vcdr = get_vcdr(test_label)
        mae = abs(predict_vcdr - test_vcdr)
        sum_mae += mae

        od_predict, oc_predict = get_oc_od(predict_img)
        od_label, oc_label = get_oc_od(test_label)

        od_iou = get_iou(od_label, od_predict)
        sum_od_iou += od_iou
        oc_iou = get_iou(oc_label, oc_predict)
        sum_oc_iou += oc_iou

        od_dice = get_dice(od_label, od_predict)
        sum_od_dice += od_dice
        oc_dice = get_dice(oc_label, oc_predict)
        sum_oc_dice += oc_dice

        print(img_name + '\tmae:' + str(mae) + '\tod_iou:' + str(od_iou) + '\toc_iou:' + str(oc_iou) +
              '\tod_dice:' + str(od_dice) + '\toc_dice:' + str(oc_dice))

    print('sum_mae:' + str(sum_mae / len(predict_img_file_list)) +
          '\tsum_od_iou:' + str(sum_od_iou / len(predict_img_file_list)) +
          '\tsum_oc_iou:' + str(sum_oc_iou / len(predict_img_file_list)) +
          '\tsum_od_dice:' + str(sum_od_dice / len(predict_img_file_list)) +
          '\tsum_oc_dice:' + str(sum_oc_dice / len(predict_img_file_list)))


if __name__ == '__main__':
    predict_img_file_path = './datasets/refuge_datasets/test/predict/'
    test_label_file_path = './datasets/refuge_datasets/test/label/'
    get_metrics(predict_img_file_path, test_label_file_path)