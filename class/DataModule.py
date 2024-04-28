import sys
from collections import defaultdict
import numpy as np
import random
from math import sqrt
import scipy.sparse as sp
from tqdm import tqdm

class DataModule():
    def __init__(self, conf, filename):
        self.conf = conf
        self.data_dict = {}
        self.terminal_flag = 1
        self.filename = filename
        self.index = 0

###########################################  Initalize Procedures ############################################
    def prepareModelSupplement(self, model):
        data_dict = {}
        if 'CONSUMED_ITEMS_SPARSE_MATRIX' in model.supply_set:
            self.generateConsumedItemsSparseMatrix()
            data_dict['CONSUMED_ITEMS_INDICES_INPUT'] = self.consumed_items_indices_list
            data_dict['CONSUMED_ITEMS_VALUES_INPUT'] = self.consumed_items_values_list
            data_dict['CONSUMED_ITEMS_VALUES_ONES_INPUT'] = self.consumed_items_values_ones_list
            data_dict['JOINT_VALUES_INPUT'] = self.joint_values_list
        if 'SOCIAL_NEIGHBORS_SPARSE_MATRIX' in model.supply_set:
            if self.conf.data_name in ['flickr', 'ciao']:
                self.readSocialNeighbors(friends_flag=0)
            else:
                self.readSocialNeighbors(friends_flag=1)
            self.generateSocialNeighborsSparseMatrix()
            data_dict['SOCIAL_NEIGHBORS_INDICES_INPUT'] = self.social_neighbors_indices_list
            data_dict['SOCIAL_NEIGHBORS_VALUES_INPUT'] = self.social_neighbors_values_list
            data_dict['SOCIAL_NEIGHBORS_VALUES_ONES_INPUT'] = self.social_neighbors_values_ones_list
        if 'KD_SOCIAL_NEIGHBORS_SPARSE_MATRIX' in model.supply_set:
            data_dict['KD_SOCIAL_NEIGHBORS_INDICES_INPUT'] = self.kd_social_neighbors_indices_list
            data_dict['KD_SOCIAL_NEIGHBORS_VALUES_INPUT'] = self.kd_social_neighbors_values_list


        return data_dict



    def initializeRankingTrain(self):
        self.readData()
        self.arrangePositiveData()
        self.generateTrainNegative()


    def initializeRankingVT(self):
        self.readData()
        self.arrangePositiveData()
        self.generateTrainNegative()


    def initalizeRankingEva(self):
        self.readData()
        self.arrangePositiveData()
        self.arrangeRatedData()

    def linkedMap(self):
        self.data_dict['USER_LIST'] = self.user_list
        self.data_dict['ITEM_LIST'] = self.item_list
        self.data_dict['NEG_LIST'] = self.neg_list
        mode = self.filename.strip().split('/')[-1].split('.')[1]
        if mode == 'train':
            if self.conf.model_name == 'kd_trustmf':
                self.data_dict['TRUSTER_LIST'] = self.truster_list
                self.data_dict['TRUSTEE_LIST'] = self.trustee_list
                self.data_dict['TRUST_LABEL'] = self.trust_label


    def linkedSslMap(self):
        self.data_dict['USER_LIST'] = self.user_list
        self.data_dict['ITEM_LIST'] = self.item_list
        self.data_dict['NEG_LIST'] = self.neg_list
    
    def linkedRankingEvaMap(self):
        self.data_dict['EVA_USER_LIST'] = self.eva_user_list

###########################################  Ranking ############################################
    def readData(self):
        f = open(self.filename) ## May should be specific for different subtasks
        total_user_list = set()
        hash_data = defaultdict(int)
        for _, line in enumerate(f):
            arr = line.split("\t")
            hash_data[(int(arr[0]), int(arr[1]))] = 1
            total_user_list.add(int(arr[0]))
        self.total_user_list = list(total_user_list)
        self.hash_data = hash_data
    
    def arrangePositiveData(self):
        positive_data = defaultdict(set)
        positive_item_user = defaultdict(set)
        hash_data = self.hash_data
        self.u_i_index = dict()
        for id, (u, i) in enumerate(hash_data):
            positive_data[u].add(i)
            positive_item_user[i].add(u)
            self.u_i_index[(u,i)] = id
        self.positive_data = positive_data
        self.positive_item_user = positive_item_user


    def arrangeRatedData(self):
        mode = self.filename.strip().split('/')[-1].split('.')[1]
        rated_data = defaultdict(set)
        data_dir = self.conf.data_dir
        train = "%s/%s.train.rating" % (data_dir, self.conf.data_name)
        val = "%s/%s.val.rating" % (data_dir, self.conf.data_name)
        test = "%s/%s.test.rating" % (data_dir, self.conf.data_name)
        file_list = []
        if mode == 'val':
            file_list = [train, test]
        if mode == 'test':
            file_list = [train, val]
        for file in file_list:
            f = open(file)
            for _, line in enumerate(f):
                arr = line.split('\t')
                rated_data[int(arr[0])].add(int(arr[1]))
        self.rated_data = rated_data

    '''
        This function designes for the train/val/test negative generating section
    '''
    def generateTrainNegative(self):
        num_items = self.conf.num_items
        num_negatives = self.conf.num_negatives
        hash_data = self.hash_data
        train_hash_data = list()
        neg_item_list = list()
        for (u, i) in hash_data:
            for _ in range(num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in hash_data:
                    j = np.random.randint(num_items)
                train_hash_data.append((u, i, j))
                neg_item_list.append((u,j))
        self.train_hash_data = train_hash_data
        random.shuffle(self.train_hash_data)
        self.terminal_flag = 1


    '''
        This function designes for the val/test section, compute loss
    '''
    def getVTRankingOneBatch(self):
        hash_data = self.train_hash_data
        user_list, item_list, neg_list = [], [], []
        for d in hash_data:
            user_list.append(d[0])
            item_list.append(d[1])
            neg_list.append(d[2])
        
        self.user_list = np.reshape(user_list, [-1, 1])
        self.item_list = np.reshape(item_list, [-1, 1])
        self.neg_list = np.reshape(neg_list, [-1, 1])
    
    '''
        This function designes for the training process
    '''
    def getTrainRankingBatch(self):
        train_hash_data = self.train_hash_data
        index = self.index
        batch_size = self.conf.batch_size * self.conf.num_negatives
        user_list, item_list, neg_list = [], [], []
        truster_list, trustee_list, trust_label = [], [], []
        social_neighbors = self.social_neighbors
        if index + batch_size < len(train_hash_data):
            target_data = train_hash_data[index:index+batch_size]
            self.index = index + batch_size
        else:
            target_data = train_hash_data[index:len(train_hash_data)]
            self.index = 0
            self.terminal_flag = 0
        for id, d in enumerate(target_data):
            user_list.append(d[0])
            item_list.append(d[1])
            neg_list.append(d[2])
            if self.conf.model_name == 'kd_trustmf':
                for v in social_neighbors[d[0]]:
                    truster_list.append(d[0])
                    trustee_list.append(v)
                    trust_label.append(1.0)

        self.user_list = np.reshape(user_list, [-1, 1])
        self.item_list = np.reshape(item_list, [-1, 1])
        self.neg_list = np.reshape(neg_list, [-1, 1])
        if self.conf.model_name == 'kd_trustmf':
            self.truster_list = np.reshape(truster_list, [-1, 1])
            self.trustee_list = np.reshape(trustee_list, [-1, 1])
            self.trust_label = np.reshape(trust_label, [-1, 1])


    '''
        This function designs for the rating evaluate section, generate negative batch
    '''
    def getEvaRankingBatch(self):
        batch_size = self.conf.batch_size
        total_user_list = self.total_user_list
        index = self.index
        terminal_flag = 1
        total_users = len(total_user_list)
        user_list = []
        if index + batch_size < total_users:
            batch_user_list = total_user_list[index:index+batch_size]
            self.index = index + batch_size
        else:
            terminal_flag = 0
            batch_user_list = total_user_list[index:total_users]
            self.index = 0
        for u in batch_user_list:
            user_list.append(u)
        self.eva_user_list = np.reshape(user_list, [-1, 1])
        return batch_user_list, terminal_flag

##################################################### Supplement for Sparse Computation ############################################
    def readSocialNeighbors(self, friends_flag=1):
        social_neighbors = defaultdict(set)
        links_file = open(self.conf.links_filename)
        for _, line in enumerate(links_file):
            tmp = line.split('\t')
            u1, u2 = int(tmp[0]), int(tmp[1])
            social_neighbors[u1].add(u2)
            if friends_flag == 1:
                social_neighbors[u2].add(u1)
        self.social_neighbors = social_neighbors


    '''
        Generate Social Neighbors Sparse Matrix Indices and Values
    '''
    def generateSocialNeighborsSparseMatrix(self):
        social_neighbors = self.social_neighbors
        train_users = list(self.positive_data.keys())
        social_neighbors_indices_list = []
        social_neighbors_values_list = []
        kd_social_neighbors_indices_list = []
        kd_social_neighbors_values_list = []
        social_neighbors_values_ones_list = []
        social_neighbors_dict = defaultdict(list)
        for u in social_neighbors:
            social_neighbors_dict[u] = sorted(social_neighbors[u])
            
        user_list = sorted(list(social_neighbors.keys()))
        for user in user_list:
            for friend in social_neighbors_dict[user]:
                social_neighbors_indices_list.append([user, friend])
                social_neighbors_values_list.append(1.0/len(social_neighbors_dict[user]))
                kd_social_neighbors_indices_list.append([user, friend])
                kd_social_neighbors_values_list.append(1.0 / len(social_neighbors_dict[user]))
                social_neighbors_values_ones_list.append(1.0)

        for user in train_users:
            if user in user_list:
                continue
            kd_social_neighbors_indices_list.append([user, user])
            kd_social_neighbors_values_list.append(1.0)
        self.social_neighbors_indices_list = np.array(social_neighbors_indices_list).astype(np.int64)
        self.social_neighbors_values_list = np.array(social_neighbors_values_list).astype(np.float32)
        self.kd_social_neighbors_indices_list = np.array(kd_social_neighbors_indices_list).astype(np.int64)
        self.kd_social_neighbors_values_list = np.array(kd_social_neighbors_values_list).astype(np.float32)
        self.social_neighbors_values_ones_list = np.array(social_neighbors_values_ones_list).astype(np.float32)
    '''
        Generate Consumed Items Sparse Matrix Indices and Values
    '''
    def generateConsumedItemsSparseMatrix(self):
        positive_data = self.positive_data
        positive_item_user = self.positive_item_user
        consumed_items_indices_list = []
        consumed_items_values_list = []
        consumed_items_values_ones_list = []
        consumed_items_dict = defaultdict(list)
        joint_values_list = []
        for u in positive_data:
            consumed_items_dict[u] = sorted(positive_data[u])
        user_list = sorted(list(positive_data.keys()))
        for u in user_list:
            for i in consumed_items_dict[u]:
                consumed_items_indices_list.append([u, i])
                consumed_items_values_list.append(1.0/len(consumed_items_dict[u]))
                consumed_items_values_ones_list.append(1.0)
                joint_values_list.append(
                    1.0 / sqrt(len(positive_data[u])) / sqrt(len(positive_item_user[i])))
        self.consumed_items_indices_list = np.array(consumed_items_indices_list).astype(np.int64)
        self.consumed_items_values_list = np.array(consumed_items_values_list).astype(np.float32)
        self.consumed_items_values_ones_list = np.array(consumed_items_values_ones_list).astype(np.float32)
        self.joint_values_list = np.array(joint_values_list).astype(np.float32)