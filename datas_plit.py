import matplotlib.pyplot as plt
from collections import defaultdict
import random
import os
import sys
from scipy.io import loadmat

random.seed(1234)


#####Yelp and Flickr
def splitwholedata():
    data_name = 'flickr'
    save_dir = "../data/flickr/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data_dir = "../../origin_data/%s/" % data_name
    train = "%s/%s.train.rating" % (data_dir, data_name)
    val = "%s/%s.val.rating" % (data_dir, data_name)
    test = "%s/%s.test.rating" % (data_dir, data_name)
    link = "%s/%s.links" % (data_dir, data_name)
    # item_path = "%s/item_vector.npy" % (data_dir)
    f = open(link)
    social_hash_data = list()
    for _, line in enumerate(f):
        arr = line.split('\t')
        u1, u2 = int(arr[0]), int(arr[1])
        social_hash_data.append((u1, u2))
    social_hash_data = list(set(social_hash_data))
    file_list = [train, val, test]
    hash_data = list()
    for file in file_list:
        f = open(file)
        for _, line in enumerate(f):
            arr = line.split('\t')
            u, i = int(arr[0]), int(arr[1])
            hash_data.append((u,i))
    hash_data = list(set(hash_data))
    random.shuffle(hash_data)
    L = len(hash_data)
    train_data = hash_data[:int(0.8*L)]
    val_data = hash_data[int(0.8*L):int(0.9*L)]
    test_data = hash_data[int(0.9*L):]
    alluser, allitem = set(), set()
    for data in train_data:
        u, i = data
        alluser.add(u)
        allitem.add(i)
    alluser = list(alluser)
    allitem = list(allitem)
    print("========user length: {}========".format(len(alluser)))
    print("========item length: {}========".format(len(allitem)))
    user2idx = dict()
    item2idx = dict()
    for user in alluser:
        if user not in user2idx.keys():
            user2idx[user] = len(user2idx)
    for item in allitem:
        if item not in item2idx.keys():
            item2idx[item] = len(item2idx)
    num = 0
    with open(save_dir + '/%s.train.rating' % data_name, 'w', encoding='utf-8') as f:
        for data in train_data:
            u, i = data
            num += 1
            f.write((str(user2idx[u]) + '\t' + str(item2idx[i]) + '\t' + str(int(1.0)) + '\n'))
    print('train total ratings:%d' % num)
    f.close()
    num = 0
    with open(save_dir + '/%s.val.rating' % data_name, 'w', encoding='utf-8') as f:
        for data in val_data:
            u, i = data
            if u not in user2idx or i not in item2idx:
                continue
            num += 1
            f.write((str(user2idx[u]) + '\t' + str(item2idx[i]) + '\t' + str(int(1.0)) + '\n'))
    f.close()
    print('val total ratings:%d' % num)
    num = 0
    with open(save_dir + '/%s.test.rating' % data_name, 'w', encoding='utf-8') as f:
        for data in test_data:
            u, i = data
            if u not in user2idx or i not in item2idx:
                continue
            num += 1
            f.write((str(user2idx[u]) + '\t' + str(item2idx[i]) + '\t' + str(int(1.0)) + '\n'))
    f.close()
    print('test total ratings:%d' % num)
    num = 0
    with open(save_dir + '/%s.links' % data_name, 'w', encoding='utf-8') as f:
        for data in social_hash_data:
            u1, u2 = data
            if u1 not in user2idx or u2 not in user2idx:
                continue
            num += 1
            f.write((str(user2idx[u1]) + '\t' + str(user2idx[u2]) + '\t' + str(int(1.0)) + '\n'))
    f.close()
    print('social  total links:%d' % num)
# splitwholedata()

#####Ciao ####
def splitciao():
    data_path = '../../origin_data/ciao/'
    filename = data_path + 'rating.mat'
    image = loadmat(filename)
    rating_data = image['rating']
    print(type(rating_data))
    print(rating_data.shape)
    ### shooping websites
    ### userid, productid, categoryid, rating
    users = set()
    items = set()
    genres = set()
    ratings = set()
    helpful = set()
    r_hash_data = list()
    g_data = dict()
    r_data = defaultdict(set)
    for data in rating_data:
        u, i, g, r, h = data
        users.add(u)
        items.add(i)
        genres.add(g)
        ratings.add(r)
        helpful.add(h)
        r_hash_data.append((u, i, r))
        r_data[u].add(i)
        g_data[i] = g
    print('the ratings contains %d users, %d items, %d genres, %d ratings, %d helpful' %
          (len(users), len(items), len(genres), len(ratings), len(helpful)))
    print('ratings range', ratings)
    print('helpful range', helpful)
    with open(data_path + 'ratings.txt', 'w', encoding='utf-8') as f:
        for da in r_hash_data:
            u, i, r = da
            f.write((str(u) + '\t' + str(i) + '\t' + str(r) + '\n'))
    f.close()

    filename = data_path + '/trustnetwork.mat'
    image = loadmat(filename)
    linking_data = image['trustnetwork']
    print(type(linking_data))
    print(linking_data.shape)
    ### social network data, one way
    ### user1id, user2id
    social_user = set()
    l_hash_data = list()
    l_data = defaultdict(set)
    for data in linking_data:
        u1, u2 = data
        social_user.add(u1)
        social_user.add(u2)
        l_hash_data.append((u1, u2))
        l_data[u1].add(u2)
    print('the social data contains %d social user' % (len(social_user)))
    with open(data_path + 'trusts.txt', 'w', encoding='utf-8') as f:
        for da in l_hash_data:
            u1, u2 = da
            f.write((str(u1) + '\t' + str(u2) + '\t' + str(1.0) + '\n'))
    f.close()
    ###*************************************************************###
    data_name = 'ciao'
    save_dir = "../data/ciao/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data_dir = "../../origin_data/%s" % data_name
    rating = "%s/ratings.txt" % (data_dir)
    link = "%s/trusts.txt" % (data_dir)
    f = open(link)
    social_hash_data = list()
    user1, user2 = set(), set()
    num = 0
    for id, line in enumerate(f):
        arr = line.strip().split('\t')
        u1, u2 = int(arr[0]), int(arr[1])
        user1.add(u1)
        user2.add(u2)
        # if (u1, u2) in social_hash_data or (u2, u1) in social_hash_data:
        #     num += 1
        #     continue
        social_hash_data.append((u1, u2))
    print(len(user1), len(user2))
    print(len(social_hash_data), num)
    social_hash_data = list(set(social_hash_data))
    hash_data = list()
    f = open(rating)
    user, item = set(), set()
    user_item = defaultdict(set)
    rating_data = list()
    for id, line in enumerate(f):
        arr = line.strip().split('\t')
        u, i, r = int(arr[0]), int(arr[1]), int(arr[2])
        rating_data.append((u,i,r))
        if r < 4:
            continue
        user.add(u)
        item.add(i)
        user_item[u].add(i)
        hash_data.append((u, i))
    hash_data = list(set(hash_data))
    print(len(user), len(item))
    print(len(rating_data))
    print(len(hash_data))
    user_count = dict()
    for u, items in user_item.items():
        user_count[u] = len(items)
    conut_num = dict()
    for k, v in user_count.items():
        conut_num[v] = conut_num.get(v, 0) + 1
    print(conut_num)
    random.shuffle(hash_data)
    L = len(hash_data)
    train_data = hash_data[:int(0.8 * L)]
    val_data = hash_data[int(0.8 * L):int(0.9 * L)]
    test_data = hash_data[int(0.9 * L):]
    alluser, allitem = set(), set()
    for data in train_data:
        u, i = data
        alluser.add(u)
        allitem.add(i)
    alluser = list(alluser)
    allitem = list(allitem)
    print("========user length: {}========".format(len(alluser)))
    print("========item length: {}========".format(len(allitem)))
    user2idx = dict()
    item2idx = dict()
    for user in alluser:
        if user not in user2idx.keys():
            user2idx[user] = len(user2idx)
    for item in allitem:
        if item not in item2idx.keys():
            item2idx[item] = len(item2idx)
    num = 0
    with open(save_dir + '/%s.train.rating' % data_name, 'w', encoding='utf-8') as f:
        for data in train_data:
            u, i = data
            num += 1
            f.write((str(user2idx[u]) + '\t' + str(item2idx[i]) + '\t' + str(int(1.0)) + '\n'))
    print('train total ratings:%d' % num)
    f.close()
    num = 0
    with open(save_dir + '/%s.val.rating' % data_name, 'w', encoding='utf-8') as f:
        for data in val_data:
            u, i = data
            if u not in user2idx or i not in item2idx:
                continue
            num += 1
            f.write((str(user2idx[u]) + '\t' + str(item2idx[i]) + '\t' + str(int(1.0)) + '\n'))
    f.close()
    print('val total ratings:%d' % num)
    num = 0
    with open(save_dir + '/%s.test.rating' % data_name, 'w', encoding='utf-8') as f:
        for data in test_data:
            u, i = data
            if u not in user2idx or i not in item2idx:
                continue
            num += 1
            f.write((str(user2idx[u]) + '\t' + str(item2idx[i]) + '\t' + str(int(1.0)) + '\n'))
    f.close()
    print('test total ratings:%d' % num)
    num = 0
    with open(save_dir + '/%s.links' % data_name, 'w', encoding='utf-8') as f:
        for data in social_hash_data:
            u1, u2 = data
            if u1 not in user2idx or u2 not in user2idx:
                continue
            num += 1
            f.write((str(user2idx[u1]) + '\t' + str(user2idx[u2]) + '\t' + str(int(1.0)) + '\n'))
    f.close()
    print('social  total links:%d' % num)
# splitciao()







