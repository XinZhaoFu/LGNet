import yaml


class ConfigReader:
    def __init__(self, config_path='../config/config.yml'):
        self.config_path = config_path

        file_reader = open(self.config_path, 'r', encoding='utf-8')
        self.content_dictionary = yaml.load(file_reader.read(), Loader=yaml.FullLoader)

    def get_refuge_data(self):
        refuge_data = self.content_dictionary['refuge_data']
        print('[INFO]读取refuge数据：' + refuge_data)

        return refuge_data

