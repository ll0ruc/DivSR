from community import community_louvain
import networkx as nx
import time

def community_detection(data_name):
    social_data_path = '../data/%s/%s.links' % (data_name, data_name)
    G = nx.Graph()
    f = open(social_data_path)
    edges = set()
    nodes = set()
    st = time.time()
    for id, line in enumerate(f):
        arr = line.split('\t')
        nodes.add(int(arr[0]))
        nodes.add(int(arr[1]))
        edges.add((int(arr[0]), int(arr[1])))
        edges.add((int(arr[1]), int(arr[0])))
    print("======read linking data cost {} mins======".format((time.time() - st) / 60))
    st = time.time()
    G.add_nodes_from(list(nodes))
    print("======add nodes cost {} mins======".format((time.time() - st) / 60))
    st = time.time()
    G.add_edges_from(list(edges))
    print("======add edges cost {} mins======".format((time.time() - st) / 60))
    st = time.time()
    partition = community_louvain.best_partition(G)
    print("======community detection cost {} mins======".format((time.time() - st) / 60))
    print(len(partition))
    print('======the number of community is: ', max(partition.values()) + 1)
    with open('../data/%s/community.txt'%data_name, 'w', encoding='utf-8') as g:
        for k, v in partition.items():
            g.writelines(str(k) + ':' + str(v) + '\n')

community_detection('yelp')