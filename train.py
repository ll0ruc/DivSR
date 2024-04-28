import os
from time import time
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #ignore the warnings 


def start(conf, data, model, evaluate):
    if conf.kd == 1:
        base_name = 'kd'
    else:
        if conf.social == 1:
            base_name = 'base'
        else:
            base_name = 'without'

    result_dir = os.path.join(conf.output_dir, conf.data_name, conf.model_name, base_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # start to prepare data for training and evaluating
    data.initializeRankingHandle()
    d_train, d_val, d_test, d_val_eva, d_test_eva = data.train, data.val, data.test, data.val_eva, data.test_eva
    print('System start to load data...')
    t0 = time()
    d_train.initializeRankingTrain()
    d_val.initializeRankingVT()
    d_test.initializeRankingVT()
    d_val_eva.initalizeRankingEva()
    d_test_eva.initalizeRankingEva()
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))

    # prepare model necessary data.
    t0 = time()
    data_dict = d_train.prepareModelSupplement(model)
    model.inputSupply(data_dict)
    t1 = time()
    print('Data has been prepared successfully, cost:%.4fs' % (t1 - t0))
    model.startConstructGraph()

    # standard tensorflow running environment initialize
    tf_conf = tf.ConfigProto()
    tf_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_conf)
    sess.run(model.init)
    t2 = time()
    print('Model has been initialized successfully, cost:%.4fs' % (t2 - t1))
    print('Following will output the evaluation of the model:')

    # Start Training !!!
    best_score = 0.0
    stop = 0
    best_embed = []
    current_test_measure = []
    for epoch in range(1, conf.epochs+1):
        # optimize model with training data and compute train loss
        tmp_train_loss = []
        t0 = time()
        while d_train.terminal_flag:
            d_train.getTrainRankingBatch()
            d_train.linkedMap()
            train_feed_dict = {}
            for (key, value) in model.map_dict['train'].items():
                train_feed_dict[key] = d_train.data_dict[value]

            [sub_train_loss, _] = sess.run( \
                [model.map_dict['out']['train'], model.opt], feed_dict=train_feed_dict)
            tmp_train_loss.append(sub_train_loss)
        train_loss = np.mean(tmp_train_loss)
        embedout = sess.run(model.embedout)
        d_val.getVTRankingOneBatch()
        d_val.linkedMap()
        val_feed_dict = {}
        for (key, value) in model.map_dict['val'].items():
            val_feed_dict[key] = d_val.data_dict[value]
        val_loss = sess.run([model.map_dict['out']['val']], feed_dict=val_feed_dict)

        d_test.getVTRankingOneBatch()
        d_test.linkedMap()
        test_feed_dict = {}
        for (key, value) in model.map_dict['test'].items():
            test_feed_dict[key] = d_test.data_dict[value]
        test_loss = sess.run(model.map_dict['out']['test'], feed_dict=test_feed_dict)
        t2 = time()

        # start evaluate model performance, hr and ndcg
        def getEvaPredictions(data):
            eva_predictions = {}
            terminal_flag = 1
            while terminal_flag:
                batch_user_list, terminal_flag = data.getEvaRankingBatch()
                data.linkedRankingEvaMap()
                eva_feed_dict = {}
                for (key, value) in model.map_dict['eva'].items():
                    eva_feed_dict[key] = data.data_dict[value]
                index = 0
                tmp_negative_predictions = np.reshape(
                    sess.run(
                        model.map_dict['out']['eva'],
                        feed_dict=eva_feed_dict
                    ), [-1, conf.num_items])
                for u in batch_user_list:
                    eva_predictions[u] = tmp_negative_predictions[index]
                    index = index + 1
            return eva_predictions
        tt2 = time()
        rated_data = d_val_eva.rated_data
        true_data = d_val_eva.positive_data
        eva_predictions = getEvaPredictions(d_val_eva)
        d_val_eva.index = 0 # !!!important, prepare for new batch
        val_recall, val_ndcg, val_Rec = evaluate.evaluateRankingPerformance_fullsort(eva_predictions, rated_data, true_data, conf.topk)

        true_data = d_val_eva.positive_data
        item_user = d_train.positive_item_user
        val_recall, val_ndcg, val_aggdiv, val_entory = evaluate.rankingMeasure(val_Rec, true_data, item_user)
        tt3 = time()
        print('Epoch:%d, compute loss cost:%.4fs, train loss:%.4f, val loss:%.4f, test loss:%.4f' % \
            (epoch, (t2-t0), train_loss, val_loss[0], test_loss))
        print('Evaluate val cost:%.4fs, recall:%.4f, ndcg:%.4f, aggdiv:%.5f, entory:%.5f' % (
            (tt3-tt2), val_recall, val_ndcg, val_aggdiv, val_entory))

        d_train.generateTrainNegative()
        stop += 1
        if val_recall > best_score:
            best_score = val_recall
            best_embed = embedout
            stop = 0
            tt4 = time()
            rated_data = d_test_eva.rated_data
            true_data = d_test_eva.positive_data
            eva_predictions = getEvaPredictions(d_test_eva)
            d_test_eva.index = 0
            recall, ndcg, Reclist = evaluate.evaluateRankingPerformance_fullsort(eva_predictions, rated_data,
                                                                                 true_data,
                                                                                 conf.topk)
            true_data = d_test_eva.positive_data
            item_user = d_train.positive_item_user
            recall, ndcg, aggdiv, entory = evaluate.rankingMeasure(Reclist, true_data, item_user)
            current_test_measure = [epoch, Reclist]
            tt5 = time()
            print('Evaluate test cost:%.4fs, recall:%.4f, ndcg:%.4f, aggdiv:%.5f, entory:%.5f' % (
                (tt5 - tt4), recall, ndcg, aggdiv, entory))

        if stop == conf.patience or epoch == conf.epochs:
            true_data = d_test_eva.positive_data
            item_user = d_train.positive_item_user
            best_epoch, Reclist = current_test_measure
            recall, ndcg, aggdiv, entory = evaluate.rankingMeasure(Reclist, true_data, item_user)
            fileName = str(conf.social) + '-' + str(conf.seed) + '.txt'
            with open(os.path.join(result_dir, fileName), 'w', encoding='utf-8') as g:
                for k, v in Reclist.items():
                    g.writelines(str(k) + ':' + str(v) + '\n')
            g.close()
            best_user_embed, best_item_embed = best_embed
            np.save(os.path.join(result_dir, "%s-%s-user_embed.npy"% (str(conf.social),str(conf.seed))), best_user_embed)
            np.save(os.path.join(result_dir, "%s-%s-item_embed.npy" % (str(conf.social), str(conf.seed))), best_item_embed)
            print("Early stop at epoch:%d, best test performance, epoch:%d, recall:%.5f, ndcg:%.5f, aggdiv:%.5f, entory:%.5f" %
                  (epoch, best_epoch, recall, ndcg, aggdiv, entory))

            break
