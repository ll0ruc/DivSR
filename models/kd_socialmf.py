import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from models.layer import KD_Loss

class kd_socialmf():
    def __init__(self, conf):
        self.conf = conf
        self.supply_set = (
            'SOCIAL_NEIGHBORS_SPARSE_MATRIX',
            'CONSUMED_ITEMS_SPARSE_MATRIX',
            'KD_SOCIAL_NEIGHBORS_SPARSE_MATRIX'
        )

    def startConstructGraph(self):
        self.initializeNodes()
        self.constructTrainGraph()
        self.saveVariables()
        self.defineMap()

    def inputSupply(self, data_dict):
        self.social_neighbors_indices_input = data_dict['SOCIAL_NEIGHBORS_INDICES_INPUT']
        self.social_neighbors_values_input = data_dict['SOCIAL_NEIGHBORS_VALUES_INPUT']
        self.consumed_items_indices_input = data_dict['CONSUMED_ITEMS_INDICES_INPUT']
        self.consumed_items_values_input = data_dict['CONSUMED_ITEMS_VALUES_INPUT']
        self.kd_social_neighbors_indices_input = data_dict['KD_SOCIAL_NEIGHBORS_INDICES_INPUT']
        self.kd_social_neighbors_values_input = data_dict['KD_SOCIAL_NEIGHBORS_VALUES_INPUT']
        self.social_neighbors_dense_shape = np.array([self.conf.num_users, self.conf.num_users]).astype(np.int64)
        self.consumed_items_dense_shape = np.array([self.conf.num_users, self.conf.num_items]).astype(np.int64)
        self.kd_social_neighbors_dense_shape = np.array([self.conf.num_users, self.conf.num_users]).astype(np.int64)


        self.social_neighbors_sparse_matrix = tf.SparseTensor(
            indices = self.social_neighbors_indices_input, 
            values = self.social_neighbors_values_input,
            dense_shape=self.social_neighbors_dense_shape
        )
        self.consumed_items_sparse_matrix = tf.SparseTensor(
            indices = self.consumed_items_indices_input, 
            values = self.consumed_items_values_input,
            dense_shape=self.consumed_items_dense_shape
        )
        self.kd_social_neighbors_sparse_matrix = tf.SparseTensor(
            indices=self.kd_social_neighbors_indices_input,
            values=self.kd_social_neighbors_values_input,
            dense_shape=self.kd_social_neighbors_dense_shape
        )

    def generateUserEmbeddingFromSocialNeighbors(self, current_user_embedding):
        user_embedding_from_social_neighbors = tf.sparse_tensor_dense_matmul(
            self.social_neighbors_sparse_matrix, current_user_embedding
        )
        return user_embedding_from_social_neighbors


    def generateUserEmebddingFromConsumedItems(self, current_item_embedding):
        user_embedding_from_consumed_items = tf.sparse_tensor_dense_matmul(
            self.consumed_items_sparse_matrix, current_item_embedding
        )
        return user_embedding_from_consumed_items

    def initializeNodes(self):
        self.item_input = tf.placeholder("int32", [None, 1]) # Get item embedding from the core_item_input
        self.user_input = tf.placeholder("int32", [None, 1]) # Get user embedding from the core_user_input
        self.neg_input = tf.placeholder("int32", [None, 1])

        self.user_embedding = tf.Variable(
            tf.random_normal([self.conf.num_users, self.conf.dimension], stddev=0.01), name='user_embedding')
        self.item_embedding = tf.Variable(
            tf.random_normal([self.conf.num_items, self.conf.dimension], stddev=0.01), name='item_embedding')
        if self.conf.kd == 1:
            self.user_emb_0 = tf.constant(np.load(self.conf.embed_dir), dtype=tf.float32)

    def constructTrainGraph(self):
        print('Current model is %s' % self.conf.model_name)
        item_embedding = self.item_embedding
        user_embedding = self.user_embedding
        user_embedding_from_social = self.generateUserEmbeddingFromSocialNeighbors(user_embedding)
        self.final_user_embedding = user_embedding
        self.final_item_embedding = item_embedding
        self.embedout = [self.final_user_embedding, self.final_item_embedding]
        latest_user_latent = tf.gather_nd(self.final_user_embedding, self.user_input)
        latest_item_latent = tf.gather_nd(self.final_item_embedding, self.item_input)
        latest_neg_latent = tf.gather_nd(self.final_item_embedding, self.neg_input)
        predict_vector = tf.multiply(latest_user_latent, latest_item_latent)
        self.prediction = tf.reduce_sum(predict_vector, 1, keepdims=True)
        self.test = tf.matmul(latest_user_latent, tf.transpose(self.final_item_embedding))
        neg_vector = tf.multiply(latest_user_latent, latest_neg_latent)
        neg_prediction = tf.reduce_sum(neg_vector, 1, keepdims=True)
        self.rec_loss = -tf.reduce_sum(tf.log(tf.sigmoid(self.prediction - neg_prediction) + 1e-6))
        init_user_latent = tf.gather_nd(self.user_embedding, self.user_input)
        init_item_latent = tf.gather_nd(self.item_embedding, self.item_input)
        init_neg_latent = tf.gather_nd(self.item_embedding, self.neg_input)
        self.reg_loss = self.conf.regu * (
                    tf.nn.l2_loss(init_user_latent) + tf.nn.l2_loss(init_item_latent) + tf.nn.l2_loss(init_neg_latent))
        self.opt_loss = self.rec_loss + self.reg_loss
        if self.conf.social == 1:
            self.opt_loss += self.conf.regt * tf.nn.l2_loss(latest_user_latent - tf.gather_nd(user_embedding_from_social, self.user_input))
        if self.conf.kd == 1:
            kd_loss = KD_Loss(self.final_user_embedding, self.user_emb_0, self.user_input, self.kd_social_neighbors_sparse_matrix)
            self.opt_loss += self.conf.gamma * kd_loss

        self.opt = tf.train.AdamOptimizer(self.conf.learning_rate).minimize(self.opt_loss)
        self.init = tf.global_variables_initializer()

    def saveVariables(self):
        ############################# Save Variables #################################
        variables_dict = {}
        variables_dict[self.user_embedding.op.name] = self.user_embedding
        variables_dict[self.item_embedding.op.name] = self.item_embedding
        self.saver = tf.train.Saver(variables_dict)
        ############################# Save Variables #################################
    
    def defineMap(self):
        map_dict = {}
        map_dict['train'] = {
            self.user_input: 'USER_LIST', 
            self.item_input: 'ITEM_LIST', 
            self.neg_input: 'NEG_LIST'
        }
        
        map_dict['val'] = {
            self.user_input: 'USER_LIST', 
            self.item_input: 'ITEM_LIST',
            self.neg_input: 'NEG_LIST'
        }

        map_dict['test'] = {
            self.user_input: 'USER_LIST', 
            self.item_input: 'ITEM_LIST',
            self.neg_input: 'NEG_LIST'
        }

        map_dict['eva'] = {
            self.user_input: 'EVA_USER_LIST'
        }

        map_dict['out'] = {
            'train': self.rec_loss,
            'val': self.rec_loss,
            'test': self.rec_loss,
            'eva': self.test
        }

        self.map_dict = map_dict
