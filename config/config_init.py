import yaml


def config_init():
    init_data = {
        'refuge_data': {
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
        }
    }
    file_writer = open('./config.yml', 'w', encoding='utf-8')
    yaml.dump(init_data, file_writer, allow_unicode=True)
    yaml.load(open('./config.yml'), Loader=yaml.FullLoader)
    file_writer.close()


if __name__ == '__main__':
    config_init()
