import yaml
from loguru import logger


class ConfigReader:
    def __init__(self, config_path):
        self.config_path = config_path

        file_reader = open(self.config_path, 'r', encoding='utf-8')
        self.content_dictionary = yaml.load(file_reader.read(), Loader=yaml.FullLoader)
        logger.add('./log/log_{time}.log', format='{time} | {message}', level='INFO')
        logger.info(self.content_dictionary)

        file_reader.close()

    def get_refuge_info(self):
        refuge_info = self.content_dictionary['refuge_info']
        print('[INFO]读取refuge数据')

        return refuge_info

    def get_train_info(self):
        train_info = self.content_dictionary['train_info']
        print('[INFO]读取训练信息')

        return train_info

    def get_lgnet_info(self):
        lgnet_info = self.content_dictionary['lgnet_info']
        print('[INFO]读取LGNet信息')

        return lgnet_info


