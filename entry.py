import sys, os, argparse
sys.path.append(os.path.join(os.getcwd(), 'class'))
from ParserConf import ParserConf
from DataUtil import DataUtil
from Evaluate import Evaluate
from util import init_seed
import train as starter
from models.kd_trustmf import kd_trustmf
from models.kd_socialmf import kd_socialmf
from models.kd_mhcn import kd_mhcn

def executeTrainModel(config_path, para):
    #print('System start to prepare parser config file...')
    conf = ParserConf(config_path)
    conf.parserDict(para)
    conf.parserConf()
    print('all parameters:', conf.conf_dict)
    #print('System start to load TensorFlow graph...')
    try:
        importStr = 'from models.' + conf.model_name + ' import ' + conf.model_name
        exec(importStr)
    except ImportError:
        print('===Please input a correct model name===')

    model = eval(para['model_name'])(conf)
    #print('System start to load data...')
    data = DataUtil(conf)
    evaluate = Evaluate(conf)
    starter.start(conf, data, model, evaluate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Welcome to DivSR Experiment Platform Entry')
    parser.add_argument('--data_name', default='yelp', help='data name')
    parser.add_argument('--model_name', default='kd_diffnet',help='model name')
    parser.add_argument('--output_dir', default='./results', help='output directory')
    parser.add_argument('--gpu', default='0', help='available gpu id')
    parser.add_argument('--social', default=0, type=int, help='whether use social')
    parser.add_argument('--seed', default=42, type=int, help='the random seed')
    parser.add_argument('--gamma', default=1.0, type=float, help='the weight for KD_loss')
    parser.add_argument('--kd', default=0, type=int, help='whether use KD_loss')
    parser.add_argument('--topk', default=100, type=int, help='recommendation top-k items')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--dimension', default=64, type=int, help='embedding size for all layer')
    parser.add_argument('--patience', default=40, type=int, help='early_stop validation')
    parser.add_argument('--batch_size', default=5000, type=int, help='batch size')
    parser.add_argument('--epochs', default=1000, type=int, help='epoch number')
    parser.add_argument('--regu', default=0.001, type=float, help='L2 regularization for embeddings')
    parser.add_argument('--num_negatives', default=1, type=int, help='negative sampler number for each positive pair')
    ##

    args = parser.parse_args()
    init_seed(int(args.seed))
    device_id = str(args.gpu)
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    config_path = os.path.join(os.getcwd(), 'conf/%s.ini' % (args.data_name))
    para = vars(args)
    executeTrainModel(config_path, para)
