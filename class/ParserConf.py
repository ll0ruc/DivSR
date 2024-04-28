
import configparser as cp
import re, os

class ParserConf():

    def __init__(self, config_path):
        self.config_path = config_path
        self.conf_dict = {}

    def processValue(self, key, value):
        #print(key, value)
        tmp = value.split(' ')
        dtype = tmp[0]
        value = tmp[1:]

        if value != None:
            if dtype == 'string':
                self.conf_dict[key] = vars(self)[key] = value[0]
            elif dtype == 'int':
                self.conf_dict[key] = vars(self)[key] = int(value[0])
            elif dtype == 'float':
                self.conf_dict[key] = vars(self)[key] = float(value[0])
            elif dtype == 'list':
                self.conf_dict[key] = vars(self)[key] = [i for i in value]
            elif dtype == 'int_list':
                self.conf_dict[key] = vars(self)[key] = [int(i) for i in value]
            elif dtype == 'float_list':
                self.conf_dict[key] = vars(self)[key] = [float(i) for i in value]
        else:
            print('%s value is None' % key)

    def parserConf(self):
        conf = cp.ConfigParser()
        conf.read(self.config_path)
        self.conf = conf
        model_name = self.conf_dict['model_name']
        for section in conf.sections():
            if section == 'Basic Configure' or section == model_name:
                for (key, value) in conf.items(section):
                    self.processValue(key, value)
       
        self.data_dir = os.path.join(os.getcwd(), 'data/%s' % self.conf_dict['data_name'])
        self.links_filename = os.path.join(os.getcwd(), 'data/%s/%s.links' % (self.conf_dict['data_name'], self.conf_dict['data_name']))
        if self.conf_dict['kd'] == 1:
            seed = self.conf_dict['seed']
            self.embed_dir = os.path.join(os.getcwd(), '%s/%s/%s/without/0-%s-user_embed.npy' % (self.conf_dict['output_dir'], self.conf_dict['data_name'], model_name, seed))

    def parserDict(self, config_dict):
        for k, v in config_dict.items():
            self.conf_dict[k] = vars(self)[k] = v