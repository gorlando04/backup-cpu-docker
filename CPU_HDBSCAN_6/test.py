import numpy as np
import hdbscan
import pandas as pd
import networkx
from newick import loads
from apted import APTED
import sys

sys.setrecursionlimit(int(1e9))


def tree_to_newick(linkage_df):
    # Descobre número de pontos (folhas)
    n_points = linkage_df.shape[0] + 1

    # Mapeia id do nó para sua string Newick
    newick_map = {i: f"{i}" for i in range(n_points)}

    # Se tiver distância (ou lambda_val), use como comprimento do galho
    use_branch_length = 'lambda_val' in linkage_df.columns or 'distance' in linkage_df.columns

    # Itera em ordem de construção
    for _, row in linkage_df.iterrows():
        left = row['left_child']
        right = row['right_child']
        child = row['parent']

        # Define comprimento de galho, se aplicável
        if use_branch_length:
            branch_len = row.get('lambda_val', row.get('distance', 0))
            left_str = f"{newick_map[left]}:{branch_len:.4f}"
            right_str = f"{newick_map[right]}:{branch_len:.4f}"
        else:
            left_str = newick_map[left]
            right_str = newick_map[right]

        newick_map[child] = f"({left_str},{right_str})"

    # Último nó é a raiz
    root = linkage_df['parent'].iloc[-1]
    return newick_map[root] + ';'


import numpy as np
import hdbscan

# 1. Criar dados de exemplo
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=int(1e4), centers=5, cluster_std=12.0, random_state=42)

# 2. Rodar HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=2,algorithm='generic')
clusterer.fit(X)

brkt2 = tree_to_newick(clusterer.single_linkage_tree_.to_pandas())
tree2 = loads(brkt2)


# 2. Rodar HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=20,algorithm='prims_kdtree',allow_single_cluster=True)
clusterer.fit(X)

brkt1 = tree_to_newick(clusterer.single_linkage_tree_.to_pandas())
tree1 = loads(brkt1)


#apted = APTED(tree1[0], tree2[0])
#ted = apted.compute_edit_distance()
#print(f"TED: {ted}")
