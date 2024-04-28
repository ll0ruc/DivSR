from DataModule import DataModule

class DataUtil():
    def __init__(self, conf):
        self.conf = conf

    def initializeRankingHandle(self):
        self.createTrainHandle()
        self.createEvaluateHandle()
    
    def createTrainHandle(self):
        data_dir = self.conf.data_dir
        train_filename = "%s/%s.train.rating" % (data_dir, self.conf.data_name)
        val_filename = "%s/%s.val.rating" % (data_dir, self.conf.data_name)
        test_filename = "%s/%s.test.rating" % (data_dir, self.conf.data_name)

        self.train = DataModule(self.conf, train_filename)
        self.val = DataModule(self.conf, val_filename)
        self.test = DataModule(self.conf, test_filename)

    def createEvaluateHandle(self):
        data_dir = self.conf.data_dir
        val_filename = "%s/%s.val.rating" % (data_dir, self.conf.data_name)
        test_filename = "%s/%s.test.rating" % (data_dir, self.conf.data_name)

        self.val_eva = DataModule(self.conf, val_filename)
        self.test_eva = DataModule(self.conf, test_filename)
