import yaml
from loguru import logger


def config_init():
    init_data = {
        'refuge_info': {
            'img_original_path': '../datasets/refuge_img/',
            'label_original_path': '../datasets/refuge_label/',
            'train_img_path': '../datasets/train/img/',
            'train_label_path': '../datasets/train/label/',
            'validation_img_path': '../datasets/validation/img/',
            'validation_label_path': '../datasets/validation/label/',
            'test_img_path': '../datasets/test/img/',
            'test_label_path': '../datasets/test/label/',
            'train_file_number_rate': 0.6,
            'validation_file_number_rate': 0.2,
            'test_file_number_rate': 0.2
        },

        'train_info': {
            'experiment_name': 'lgnet_refuge',
            'is_load_weight': False,
            'batch_size': 4,
            'epochs': 1,
            'is_data_augmentation': False,
            'learning_rate': 0.0001,
            'augmentation_rate': 0,
            'checkpoint_save_path': './checkpoint/',
            'checkpoint_input_path': '',
            'optimizers': 'Adam',
            'num_class': 3,
            'model_name': 'lgnet'
        },

        'lgnet_info': {
            'filters_cbr': 32,
            'num_cbr': 1,
            'end_activation': 'softmax'
        }
    }

    logger.add('../log/config_init.log', format='{time} | {message}', level='INFO')
    logger.info(init_data)

    file_writer = open('./config.yml', 'w', encoding='utf-8')
    yaml.dump(init_data, file_writer, allow_unicode=True)
    yaml.load(open('./config.yml'), Loader=yaml.FullLoader)
    file_writer.close()


if __name__ == '__main__':
    config_init()
