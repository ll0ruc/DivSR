import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def generateNeighborEmbeddingForKd(sparse_matrix, current_embedding):
    neighbor_embedding_from_anchor = tf.sparse_tensor_dense_matmul(
        sparse_matrix, current_embedding
    )
    return neighbor_embedding_from_anchor

def KD_Loss(student_embedding, teacher_embedding, index_input, sparse_matrix):
    anchor_latent_hat = tf.gather_nd(student_embedding, index_input)
    anchor_neigh_hat = generateNeighborEmbeddingForKd(sparse_matrix, student_embedding)
    anchor_neigh_latent_hat = tf.gather_nd(anchor_neigh_hat, index_input)
    anchor_latent_hat = tf.nn.l2_normalize(anchor_latent_hat, 1)

    anchor_neigh_latent_hat = tf.nn.l2_normalize(anchor_neigh_latent_hat, 1)
    anchor_norm_hat = tf.norm(anchor_latent_hat, ord=2, axis=1, keepdims=True)
    anchor_nei_norm_hat = tf.norm(anchor_neigh_latent_hat, ord=2, axis=1, keepdims=True)
    anchor_nei_s_hat = tf.multiply(anchor_latent_hat, anchor_neigh_latent_hat)
    anchor_nei_s_hat = tf.reduce_sum(anchor_nei_s_hat, 1, keepdims=True)
    anchor_sim_hat = tf.div(anchor_nei_s_hat, anchor_norm_hat)
    anchor_sim_hat = tf.div(anchor_sim_hat, anchor_nei_norm_hat)

    anchor_neigh = generateNeighborEmbeddingForKd(sparse_matrix, teacher_embedding)
    anchor_neigh_latent = tf.gather_nd(anchor_neigh, index_input)
    anchor_latent = tf.gather_nd(teacher_embedding, index_input)
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