import networkx as nx
import itertools
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np



pre_dfRegNet = pd.read_csv('tableData.csv')  
dfRegNet = pre_dfRegNet[pd.notna(pre_dfRegNet['3)RegulatorGeneName'])] #dropping NaN 
print(dfRegNet.shape)

#DiGraph g
regnet = dfRegNet.loc[:, ['3)RegulatorGeneName', '5)regulatedName']]
G  = nx.DiGraph()
G.add_edges_from(regnet.values)


 # assuming g is your existing DiGraph

# Step 1: Extract all unique 3-node subgraphs (as node sets)
subgraph_node_sets = list(itertools.combinations(G.nodes, 3))

# Step 2: Generate all 3-node directed motifs
def generate_all_3_node_digraphs():
    motifs = []
    nodes = [0, 1, 2]
    edge_list = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    for edges in itertools.product([0, 1], repeat=6):
        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        for i, flag in enumerate(edges):
            if flag:
                g.add_edge(*edge_list[i])
        motifs.append(g)
    return motifs

motif_types = generate_all_3_node_digraphs()

# Step 3: Define a function to classify a subgraph
def classify_subgraph(nodes):
    sg = G.subgraph(nodes).copy()
    for i, motif in enumerate(motif_types):
        if nx.is_isomorphic(sg, motif): 
            return i
    return None  # should not happen unless a bug

# Step 4: Run in parallel
with ProcessPoolExecutor() as executor:
    motif_ids = list(tqdm(executor.map(classify_subgraph, subgraph_node_sets),
                          total=len(subgraph_node_sets),
                          desc="Parallel motif counting"))

# Step 5: Count and report
motif_counts = Counter(motif_ids)
for motif_id, count in motif_counts.items():
    print(f"Motif {motif_id}: {count} occurrences")
