import yaml
from loguru import logger


def config_init():
    init_data = {
        'refuge_info': {
            'img_original_path': './datasets/refuge_datasets/refuge_img/',
            'label_original_path': './datasets/refuge_datasets/refuge_label/',
            'train_img_path': './datasets/refuge_datasets/train/img/',
            'train_label_path': './datasets/refuge_datasets/train/label/',
            'validation_img_path': './datasets/refuge_datasets/validation/img/',
            'validation_label_path': './datasets/refuge_datasets/validation/label/',
            'test_img_path': './datasets/refuge_datasets/test/img/',
            'test_label_path': './datasets/refuge_datasets/test/label/',
            'train_aug_img_path': './datasets/refuge_datasets/train/aug_img/',
            'train_aug_label_path': './datasets/refuge_datasets/train/aug_label/',
            'augmentation_rate': 9
        },

        'train_info': {
            'experiment_name': 'lgnet_refuge',
            'is_load_weight': False,
            'batch_size': 4,
            'epochs': 1,
            'is_data_augmentation': True,
            'learning_rate': 0.0001,
            'checkpoint_save_path': './checkpoint/',
            'checkpoint_input_path': '',
            'optimizers': 'Adam',
            'num_class': 3,
            'model_name': 'lgnet'
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
