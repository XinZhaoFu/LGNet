import yaml
from loguru import logger


class ConfigReader:
    def __init__(self, config_path='../config/config.yml'):
        self.config_path = config_path

        file_reader = open(self.config_path, 'r', encoding='utf-8')
        self.content_dictionary = yaml.load(file_reader.read(), Loader=yaml.FullLoader)
        logger.add('../log/log_{time}.log', format='{time} | {level} | {message}', level='INFO')
        logger.info(self.content_dictionary)

        file_reader.close()

    def get_refuge_info(self):
        refuge_info = self.content_dictionary['refuge_info']
        print('[INFO]读取refuge数据')

        return refuge_info

