import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from scipy.sparse import coo_matrix


class kd_mhcn():
    def __init__(self, conf):
        self.conf = conf
        self.emb_size = self.conf.dimension
        self.graph = tf.Graph()
        self.supply_set = (
            'CONSUMED_ITEMS_SPARSE_MATRIX',
            'SOCIAL_NEIGHBORS_SPARSE_MATRIX',
            'KD_SOCIAL_NEIGHBORS_SPARSE_MATRIX'
        )

    def startConstructGraph(self):
        self.buildMotifInducedAdjacencyMatrix()
        self.initializeNodes()
        if self.conf.social == 1:
            self.constructTrainGraph()
        else:
            self.constructTrainGraphWosocial()
        self.saveVariables()
        self.defineMap()

    def inputSupply(self, data_dict):
        self.consumed_items_indices_input = data_dict['CONSUMED_ITEMS_INDICES_INPUT']
        ###  need the 1 in every entry.
        self.consumed_items_values_input = data_dict['CONSUMED_ITEMS_VALUES_ONES_INPUT']
        self.joint_users_items_values_input = data_dict['JOINT_VALUES_INPUT']
        self.social_neighbors_indices_input = data_dict['SOCIAL_NEIGHBORS_INDICES_INPUT']
        self.social_neighbors_values_input = data_dict['SOCIAL_NEIGHBORS_VALUES_ONES_INPUT']
        self.kd_social_neighbors_indices_input = data_dict['KD_SOCIAL_NEIGHBORS_INDICES_INPUT']
        self.kd_social_neighbors_values_input = data_dict['KD_SOCIAL_NEIGHBORS_VALUES_INPUT']

    def buildSparseRelationMatrix(self):
        entries = self.social_neighbors_values_input
        row, col = list(self.social_neighbors_indices_input[:, 0]), list(self.social_neighbors_indices_input[:, 1])
        AdjacencyMatrix = coo_matrix((entries, (row, col)), shape=(self.conf.num_users, self.conf.num_users),
                                     dtype=np.float32)
        return AdjacencyMatrix

    def buildSparseRatingMatrix(self):
        entries = self.consumed_items_values_input
        row, col = list(self.consumed_items_indices_input[:, 0]), list(self.consumed_items_indices_input[:, 1])
        ratingMatrix = coo_matrix((entries, (row, col)), shape=(self.conf.num_users, self.conf.num_items),
                                  dtype=np.float32)
        return ratingMatrix

    def buildJointAdjacency(self):
        indices = self.consumed_items_indices_input
        values = self.joint_users_items_values_input
        norm_adj = tf.SparseTensor(indices=indices, values=values,
                                   dense_shape=[self.conf.num_users, self.conf.num_items])
        return norm_adj

    def buildSparseKdMatrix(self):
        indices = self.kd_social_neighbors_indices_input
        values = self.kd_social_neighbors_values_input
        norm_adj = tf.SparseTensor(indices=indices, values=values,
                                   dense_shape=[self.conf.num_users, self.conf.num_users])
        return norm_adj


    def buildMotifInducedAdjacencyMatrix(self):
        S = self.buildSparseRelationMatrix()
        Y = self.buildSparseRatingMatrix()
        self.userAdjacency = Y.tocsr()
        self.itemAdjacency = Y.T.tocsr()
        B = S.multiply(S.T)
        U = S - B
        C1 = (U.dot(U)).multiply(U.T)
        A1 = C1 + C1.T
        C2 = (B.dot(U)).multiply(U.T) + (U.dot(B)).multiply(U.T) + (U.dot(U)).multiply(B)
        A2 = C2 + C2.T
        C3 = (B.dot(B)).multiply(U) + (B.dot(U)).multiply(B) + (U.dot(B)).multiply(B)
        A3 = C3 + C3.T
        A4 = (B.dot(B)).multiply(B)
        C5 = (U.dot(U)).multiply(U) + (U.dot(U.T)).multiply(U) + (U.T.dot(U)).multiply(U)
        A5 = C5 + C5.T

        A6 = (U.dot(B)).multiply(U) + (B.dot(U.T)).multiply(U.T) + (U.T.dot(U)).multiply(B)
        A7 = (U.T.dot(B)).multiply(U.T) + (B.dot(U)).multiply(U) + (U.dot(U.T)).multiply(B)
        A8 = (Y.dot(Y.T)).multiply(B)
        A9 = (Y.dot(Y.T)).multiply(U)
        A9 = A9 + A9.T
        if self.conf.social == 1:
            A10 = Y.dot(Y.T) - A8 - A9
        else:
            A10 = Y.dot(Y.T)
        H_s = sum([A1, A2, A3, A4, A5, A6, A7])
        H_s = H_s.multiply(1.0 / (H_s.sum(axis=1).reshape(-1, 1)))
        H_j = sum([A8, A9])
        H_j = H_j.multiply(1.0 / (H_j.sum(axis=1).reshape(-1, 1)))
        H_p = A10
        H_p = H_p.multiply(H_p > 1)
        H_p = H_p.multiply(1.0 / (H_p.sum(axis=1).reshape(-1, 1)))

        # initialize adjacency matrices
        H_s = self.adj_to_sparse_tensor(H_s)
        H_j = self.adj_to_sparse_tensor(H_j)
        H_p = self.adj_to_sparse_tensor(H_p)
        R = self.buildJointAdjacency()
        self.KR = self.buildSparseKdMatrix()
        # self.KR_ITEM = self.buildSparseItemNeighborMatrix()
        self.M_matrices = [H_s, H_j, H_p, R]

    def adj_to_sparse_tensor(self, adj):
        adj = adj.tocoo()
        indices = np.mat(list(zip(adj.row, adj.col)))
        adj = tf.SparseTensor(indices, adj.data.astype(np.float32), adj.shape)
        return adj

    # define inline functions
    def self_gating(self, em, channel):
        return tf.multiply(em, tf.nn.sigmoid(
            tf.matmul(em, self.weights['gating%d' % channel]) + self.weights['gating_bias%d' % channel]))

    def self_supervised_gating(self, em, channel):
        return tf.multiply(em, tf.nn.sigmoid(
            tf.matmul(em, self.weights['sgating%d' % channel]) + self.weights['sgating_bias%d' % channel]))

    def channel_attention(self, *channel_embeddings):
        weights = []
        for embedding in channel_embeddings:
            weights.append(tf.reduce_sum(
                tf.multiply(self.weights['attention'], tf.matmul(embedding, self.weights['attention_mat'])), 1))
        score = tf.nn.softmax(tf.transpose(weights))
        mixed_embeddings = 0
        for i in range(len(weights)):
            mixed_embeddings += tf.transpose(
                tf.multiply(tf.transpose(score)[i], tf.transpose(channel_embeddings[i])))
        return mixed_embeddings, score

    def initializeNodes(self):
        self.item_input = tf.placeholder("int32", [None, 1])  # Get item embedding from the core_item_input
        self.user_input = tf.placeholder("int32", [None, 1])  # Get user embedding from the core_user_input
        self.neg_input = tf.placeholder("int32", [None, 1])
        self.user_embedding = tf.Variable(
            tf.truncated_normal(shape=[self.conf.num_users, self.conf.dimension], stddev=0.005), name='user_embedding')
        self.item_embedding = tf.Variable(
            tf.truncated_normal(shape=[self.conf.num_items, self.conf.dimension], stddev=0.005), name='item_embedding')
        self.restore_user_embedding = tf.Variable(
            tf.random_normal([self.conf.num_users, self.conf.dimension], stddev=0.01), name='restore_user_embedding')
        self.restore_item_embedding = tf.Variable(
            tf.random_normal([self.conf.num_items, self.conf.dimension], stddev=0.01), name='restore_item_embedding')

        if self.conf.kd == 1:
            self.user_emb_0 = tf.constant(np.load(self.conf.embed_dir), dtype=tf.float32)

        initializer = tf.keras.initializers.glorot_normal()
        self.weights = {}
        self.n_channel = 4
        # define learnable paramters
        for i in range(self.n_channel):
            self.weights['gating%d' % (i + 1)] = tf.Variable(initializer([self.emb_size, self.emb_size]),
                                                             name='g_W_%d_1' % (i + 1))
            self.weights['gating_bias%d' % (i + 1)] = tf.Variable(initializer([1, self.emb_size]),
                                                                  name='g_W_b_%d_1' % (i + 1))
            self.weights['sgating%d' % (i + 1)] = tf.Variable(initializer([self.emb_size, self.emb_size]),
                                                              name='sg_W_%d_1' % (i + 1))
            self.weights['sgating_bias%d' % (i + 1)] = tf.Variable(initializer([1, self.emb_size]),
                                                                   name='sg_W_b_%d_1' % (i + 1))
        self.weights['attention'] = tf.Variable(initializer([1, self.emb_size]), name='at')
        self.weights['attention_mat'] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='atm')

    def constructTrainGraph(self):
        # self-gating
        print('Current model is %s' % self.conf.model_name)
        user_embeddings_c1 = self.self_gating(self.user_embedding, 1)
        user_embeddings_c2 = self.self_gating(self.user_embedding, 2)
        user_embeddings_c3 = self.self_gating(self.user_embedding, 3)
        simple_user_embeddings = self.self_gating(self.user_embedding, 4)
        all_embeddings_c1 = [user_embeddings_c1]
        all_embeddings_c2 = [user_embeddings_c2]
        all_embeddings_c3 = [user_embeddings_c3]
        all_embeddings_simple = [simple_user_embeddings]
        item_embeddings = self.item_embedding
        all_embeddings_i = [item_embeddings]
        self.ss_loss = 0
        H_s, H_j, H_p, R = self.M_matrices
        # multi-channel convolution
        for k in range(self.conf.gcn_layer):
            mixed_embedding = self.channel_attention(user_embeddings_c1, user_embeddings_c2, user_embeddings_c3)[
                                  0] + simple_user_embeddings / 2
            # Channel S
            user_embeddings_c1 = tf.sparse_tensor_dense_matmul(H_s, user_embeddings_c1)
            norm_embeddings = tf.math.l2_normalize(user_embeddings_c1, axis=1)
            all_embeddings_c1 += [norm_embeddings]
            # Channel J
            user_embeddings_c2 = tf.sparse_tensor_dense_matmul(H_j, user_embeddings_c2)
            norm_embeddings = tf.math.l2_normalize(user_embeddings_c2, axis=1)
            all_embeddings_c2 += [norm_embeddings]
            # Channel P
            user_embeddings_c3 = tf.sparse_tensor_dense_matmul(H_p, user_embeddings_c3)
            norm_embeddings = tf.math.l2_normalize(user_embeddings_c3, axis=1)
            all_embeddings_c3 += [norm_embeddings]
            # item convolution
            new_item_embeddings = tf.sparse_tensor_dense_matmul(tf.sparse.transpose(R), mixed_embedding)
            norm_embeddings = tf.math.l2_normalize(new_item_embeddings, axis=1)
            all_embeddings_i += [norm_embeddings]
            simple_user_embeddings = tf.sparse_tensor_dense_matmul(R, item_embeddings)
            all_embeddings_simple += [tf.math.l2_normalize(simple_user_embeddings, axis=1)]
            item_embeddings = new_item_embeddings
        # averaging the channel-specific embeddings
        user_embeddings_c1 = tf.reduce_sum(all_embeddings_c1, axis=0)
        user_embeddings_c2 = tf.reduce_sum(all_embeddings_c2, axis=0)
        user_embeddings_c3 = tf.reduce_sum(all_embeddings_c3, axis=0)
        simple_user_embeddings = tf.reduce_sum(all_embeddings_simple, axis=0)
        item_embeddings = tf.reduce_sum(all_embeddings_i, axis=0)

        # aggregating channel-specific embeddings
        self.final_item_embeddings = item_embeddings
        self.final_user_embeddings, self.attention_score = self.channel_attention(user_embeddings_c1,
                                                                                  user_embeddings_c2,
                                                                                  user_embeddings_c3)
        self.final_user_embeddings += simple_user_embeddings / 2
        assign1 = tf.assign(self.restore_user_embedding, self.final_user_embeddings)
        assign2 = tf.assign(self.restore_item_embedding, self.final_item_embeddings)
        self.assign = [assign1, assign2]
        # create self-supervised loss
        self.ss_loss += self.hierarchical_self_supervision(self.self_supervised_gating(self.final_user_embeddings, 1),H_s)
        self.ss_loss += self.hierarchical_self_supervision(self.self_supervised_gating(self.final_user_embeddings, 2),H_j)
        self.ss_loss += self.hierarchical_self_supervision(self.self_supervised_gating(self.final_user_embeddings, 3),H_p)
        # embedding look-up
        self.embedout = [self.final_user_embeddings, self.final_item_embeddings]
        latest_user_latent = tf.gather_nd(self.final_user_embeddings, self.user_input)
        latest_item_latent = tf.gather_nd(self.final_item_embeddings, self.item_input)
        latest_neg_latent = tf.gather_nd(self.final_item_embeddings, self.neg_input)
        init_user_latent = tf.gather_nd(self.user_embedding, self.user_input)
        init_item_latent = tf.gather_nd(self.item_embedding, self.item_input)
        init_neg_latent = tf.gather_nd(self.item_embedding, self.neg_input)
        predict_vector = tf.multiply(latest_user_latent, latest_item_latent)
        self.prediction = tf.reduce_sum(predict_vector, 1, keepdims=True)
        self.test = tf.matmul(latest_user_latent, tf.transpose(self.final_item_embeddings))
        neg_vector = tf.multiply(latest_user_latent, latest_neg_latent)
        neg_prediction = tf.reduce_sum(neg_vector, 1, keepdims=True)
        self.y_loss = -tf.reduce_sum(tf.log(tf.sigmoid(self.prediction - neg_prediction) + 10e-8))
        self.reg_loss = 0
        for key in self.weights:
            self.reg_loss += 0.001 * tf.nn.l2_loss(self.weights[key])
        # self.reg_loss += self.conf.regu * (tf.nn.l2_loss(self.user_embedding) + tf.nn.l2_loss(self.item_embedding))
        self.reg_loss += self.conf.regu * (tf.nn.l2_loss(init_user_latent) + tf.nn.l2_loss(init_item_latent)
                                        + tf.nn.l2_loss(init_neg_latent))
        self.opt_loss = self.y_loss + self.reg_loss  + self.conf.ss_rate * self.ss_loss

        if self.conf.kd == 1:
            kd_loss = self.KD_Loss(self.final_user_embeddings, self.user_emb_0, self.user_input, self.KR)
            self.opt_loss += self.conf.gamma * kd_loss
        self.opt = tf.train.AdamOptimizer(self.conf.learning_rate).minimize(self.opt_loss)
        self.init = tf.global_variables_initializer()

    def constructTrainGraphWosocial(self):
        # self-gating
        # user_embeddings_c1 = self.self_gating(self.user_embedding, 1)
        # user_embeddings_c2 = self.self_gating(self.user_embedding, 2)
        user_embeddings_c3 = self.self_gating(self.user_embedding, 3)
        simple_user_embeddings = self.self_gating(self.user_embedding, 4)
        # all_embeddings_c1 = [user_embeddings_c1]
        # all_embeddings_c2 = [user_embeddings_c2]
        all_embeddings_c3 = [user_embeddings_c3]
        all_embeddings_simple = [simple_user_embeddings]
        item_embeddings = self.item_embedding
        all_embeddings_i = [item_embeddings]
        self.ss_loss = 0
        H_s, H_j, H_p, R = self.M_matrices
        # multi-channel convolution
        for k in range(self.conf.gcn_layer):
            mixed_embedding = user_embeddings_c3 + simple_user_embeddings/2
            # Channel S
            # user_embeddings_c1 = tf.sparse_tensor_dense_matmul(H_s, user_embeddings_c1)
            # norm_embeddings = tf.math.l2_normalize(user_embeddings_c1, axis=1)
            # all_embeddings_c1 += [norm_embeddings]
            # Channel J
            # user_embeddings_c2 = tf.sparse_tensor_dense_matmul(H_j, user_embeddings_c2)
            # norm_embeddings = tf.math.l2_normalize(user_embeddings_c2, axis=1)
            # all_embeddings_c2 += [norm_embeddings]
            # Channel P
            user_embeddings_c3 = tf.sparse_tensor_dense_matmul(H_p, user_embeddings_c3)
            norm_embeddings = tf.math.l2_normalize(user_embeddings_c3, axis=1)
            all_embeddings_c3 += [norm_embeddings]
            # item convolution
            new_item_embeddings = tf.sparse_tensor_dense_matmul(tf.sparse.transpose(R), mixed_embedding)
            norm_embeddings = tf.math.l2_normalize(new_item_embeddings, axis=1)
            all_embeddings_i += [norm_embeddings]
            simple_user_embeddings = tf.sparse_tensor_dense_matmul(R, item_embeddings)
            all_embeddings_simple += [tf.math.l2_normalize(simple_user_embeddings, axis=1)]
            item_embeddings = new_item_embeddings
        # averaging the channel-specific embeddings
        # user_embeddings_c1 = tf.reduce_sum(all_embeddings_c1, axis=0)
        # user_embeddings_c2 = tf.reduce_sum(all_embeddings_c2, axis=0)
        user_embeddings_c3 = tf.reduce_sum(all_embeddings_c3, axis=0)
        simple_user_embeddings = tf.reduce_sum(all_embeddings_simple, axis=0)
        item_embeddings = tf.reduce_sum(all_embeddings_i, axis=0)

        # aggregating channel-specific embeddings
        self.final_item_embeddings = item_embeddings
        self.final_user_embeddings = user_embeddings_c3
        self.final_user_embeddings += simple_user_embeddings / 2
        assign1 = tf.assign(self.restore_user_embedding, self.final_user_embeddings)
        assign2 = tf.assign(self.restore_item_embedding, self.final_item_embeddings)
        self.assign = [assign1, assign2]
        # create self-supervised loss
        # self.ss_loss += self.hierarchical_self_supervision(self.self_supervised_gating(self.final_user_embeddings, 1),H_s)
        # self.ss_loss += self.hierarchical_self_supervision(self.self_supervised_gating(self.final_user_embeddings, 2),H_j)
        self.ss_loss += self.hierarchical_self_supervision(self.self_supervised_gating(self.final_user_embeddings, 3),H_p)
        # embedding look-up
        self.embedout = [self.final_user_embeddings, self.final_item_embeddings]
        latest_user_latent = tf.gather_nd(self.final_user_embeddings, self.user_input)
        latest_item_latent = tf.gather_nd(self.final_item_embeddings, self.item_input)
        latest_neg_latent = tf.gather_nd(self.final_item_embeddings, self.neg_input)
        init_user_latent = tf.gather_nd(self.user_embedding, self.user_input)
        init_item_latent = tf.gather_nd(self.item_embedding, self.item_input)
        init_neg_latent = tf.gather_nd(self.item_embedding, self.neg_input)
        predict_vector = tf.multiply(latest_user_latent, latest_item_latent)
        self.prediction = tf.reduce_sum(predict_vector, 1, keepdims=True)
        self.test = tf.matmul(latest_user_latent, tf.transpose(self.final_item_embeddings))
        neg_vector = tf.multiply(latest_user_latent, latest_neg_latent)
        neg_prediction = tf.reduce_sum(neg_vector, 1, keepdims=True)
        self.y_loss = -tf.reduce_sum(tf.log(tf.sigmoid(self.prediction - neg_prediction) + 10e-8))
        self.reg_loss = 0
        for key in self.weights:
            self.reg_loss += 0.001 * tf.nn.l2_loss(self.weights[key])
        self.reg_loss += self.conf.regu * (tf.nn.l2_loss(init_user_latent) + tf.nn.l2_loss(init_item_latent)
                                        + tf.nn.l2_loss(init_neg_latent))
        self.opt_loss = self.y_loss + self.reg_loss  + self.conf.ss_rate * self.ss_loss
        self.opt = tf.train.AdamOptimizer(self.conf.learning_rate).minimize(self.opt_loss)
        self.init = tf.global_variables_initializer()


    def KD_Loss(self, final_embedding, wo_embedding, index_input, KR_side):
        anchor_latent_hat = tf.gather_nd(final_embedding, index_input)
        anchor_neigh_hat = tf.sparse_tensor_dense_matmul(KR_side, final_embedding)
        anchor_neigh_latent_hat = tf.gather_nd(anchor_neigh_hat, index_input)
        anchor_latent_hat = tf.nn.l2_normalize(anchor_latent_hat, 1)
        anchor_neigh_latent_hat = tf.nn.l2_normalize(anchor_neigh_latent_hat, 1)
        anchor_norm_hat = tf.norm(anchor_latent_hat, ord=2, axis=1, keepdims=True)
        anchor_nei_norm_hat = tf.norm(anchor_neigh_latent_hat, ord=2, axis=1, keepdims=True)
        anchor_nei_s_hat = tf.multiply(anchor_latent_hat, anchor_neigh_latent_hat)
        anchor_nei_s_hat = tf.reduce_sum(anchor_nei_s_hat, 1, keepdims=True)
        anchor_sim_hat = tf.div(anchor_nei_s_hat, anchor_norm_hat)
        anchor_sim_hat = tf.div(anchor_sim_hat, anchor_nei_norm_hat)
        anchor_neigh = tf.sparse_tensor_dense_matmul(KR_side, wo_embedding)
        anchor_neigh_latent = tf.gather_nd(anchor_neigh, index_input)
        anchor_latent = tf.gather_nd(wo_embedding, index_input)
        anchor_latent = tf.nn.l2_normalize(anchor_latent, 1)
        anchor_neigh_latent = tf.nn.l2_normalize(anchor_neigh_latent, 1)
        anchor_norm = tf.norm(anchor_latent, ord=2, axis=1, keepdims=True)
        anchor_nei_norm = tf.norm(anchor_neigh_latent, ord=2, axis=1, keepdims=True)
        anchor_nei_s = tf.multiply(anchor_latent, anchor_neigh_latent)
        anchor_nei_s = tf.reduce_sum(anchor_nei_s, 1, keepdims=True)
        anchor_sim = tf.div(anchor_nei_s, anchor_norm)
        anchor_sim = tf.div(anchor_sim, anchor_nei_norm)
        kd_loss = tf.nn.l2_loss(anchor_sim_hat - anchor_sim)
        return kd_loss

    def hierarchical_self_supervision(self, em, adj):
        def row_shuffle(embedding):
            return tf.gather(embedding, tf.random.shuffle(tf.range(tf.shape(embedding)[0])))

        def row_column_shuffle(embedding):
            corrupted_embedding = tf.transpose(
                tf.gather(tf.transpose(embedding), tf.random.shuffle(tf.range(tf.shape(tf.transpose(embedding))[0]))))
            corrupted_embedding = tf.gather(corrupted_embedding,
                                            tf.random.shuffle(tf.range(tf.shape(corrupted_embedding)[0])))
            return corrupted_embedding

        def score(x1, x2):
            return tf.reduce_sum(tf.multiply(x1, x2), 1)

        user_embeddings = em
        # user_embeddings = tf.math.l2_normalize(em,1) #For Douban, normalization is needed.
        edge_embeddings = tf.sparse_tensor_dense_matmul(adj, user_embeddings)
        # Local MIM
        pos = score(user_embeddings, edge_embeddings)
        neg1 = score(row_shuffle(user_embeddings), edge_embeddings)
        neg2 = score(row_column_shuffle(edge_embeddings), user_embeddings)
        local_loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos - neg1)) - tf.log(tf.sigmoid(neg1 - neg2)))
        # Global MIM
        graph = tf.reduce_mean(edge_embeddings, 0)
        pos = score(edge_embeddings, graph)
        neg1 = score(row_column_shuffle(edge_embeddings), graph)
        global_loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos - neg1)))
        return global_loss + local_loss

    def saveVariables(self):
        ############################# Save Variables #################################
        variables_dict = {}
        variables_dict[self.restore_user_embedding.op.name] = self.restore_user_embedding
        variables_dict[self.restore_item_embedding.op.name] = self.restore_item_embedding
        self.saver = tf.train.Saver(variables_dict, max_to_keep=1)
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
            'train': self.y_loss,
            'val': self.y_loss,
            'test': self.y_loss,
            'eva': self.test
        }

        self.map_dict = map_dict
