import numpy as np
import pandas as pd
import networkx as nx


INFTY = np.inf


class UnionFind:
    def __init__(self, N):
        self.parent_arr = [-1] * (2 * N - 1)
        self.next_label = N
        self.size_arr = [1] * N + [0] * (N - 1)
        self.parent = self.parent_arr
        self.size = self.size_arr

    def union(self, m, n):
        self.size[self.next_label] = self.size[m] + self.size[n]
        self.parent[m] = self.next_label
        self.parent[n] = self.next_label
        self.size[self.next_label] = self.size[m] + self.size[n]
        self.next_label += 1

    def fast_find(self, n):
        p = n
        while self.parent_arr[n] != -1:
            n = self.parent_arr[n]
        while self.parent_arr[p] != n:
            p, self.parent_arr[p] = self.parent_arr[p], n
        return n
        
class TreeUnionFind:
    def __init__(self, size):
        self._data_arr = np.zeros((size, 2), dtype=np.intp)
        self._data_arr[:, 0] = np.arange(size)
        self._data = self._data_arr.view(np.intp).reshape((size, 2))
        self.is_component = np.ones(size, dtype=bool)

    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)

        if self._data[x_root, 1] < self._data[y_root, 1]:
            self._data[x_root, 0] = y_root
        elif self._data[x_root, 1] > self._data[y_root, 1]:
            self._data[y_root, 0] = x_root
        else:
            self._data[y_root, 0] = x_root
            self._data[x_root, 1] += 1

    def find(self, x):
        if self._data[x, 0] != x:
            self._data[x, 0] = self.find(self._data[x, 0])
            self.is_component[x] = False
        return self._data[x, 0]

    def components(self):
        return np.where(self.is_component)[0]
        
################################################################################################################
#                                                                                                              #
#                                           APPROXIMATE HDBSCAN                                                #     
#                                                                                                              #         
#                                                                                                              #     
#                                                                                                              #     
#                                                                                                              # 
################################################################################################################

class Approximate_HDBSCAN:


    def __init__(self,MST,N,mpts) -> None:
        
        self.mst_edges = self.to_tuple(MST)
        self.N = N
        self.mpts = mpts
    
    def to_tuple(self,MST):

        mst_tuple = np.zeros((len(MST.edges),3))
        index = 0

        for u, v, data in MST.edges(data=True):
            mst_tuple[index] = [u, v, data['weight']]
            index += 1

        return mst_tuple


    def label(self):
        
        result_arr = np.zeros((len(self.mst_edges), 4))

        result = result_arr
        N = self.N 
        U = UnionFind(N)

        for index in range(len(self.mst_edges)):

            a = int(self.mst_edges[index][0])
            b = int(self.mst_edges[index][1])
            delta = self.mst_edges[index][2]

            aa, bb = U.fast_find(a), U.fast_find(b)

            result[index][0] = aa
            result[index][1] = bb
            result[index][2] = delta
            result[index][3] = U.size[aa] + U.size[bb]
            U.union(aa, bb)

        return result_arr


    def bfs_from_hierarchy(self,hierarchy, bfs_root):

        dim = hierarchy.shape[0]
        max_node = 2 * dim
        num_points = max_node - dim + 1

        to_process = [bfs_root]
        result = []

        while to_process:
            result.extend(to_process)
            to_process = [x - num_points for x in to_process if x >= num_points]
            if to_process:
                to_process = hierarchy[to_process, :2].flatten().astype(np.intp).tolist()

        return result

    def condense_tree(self,hierarchy):


        root = 2 * hierarchy.shape[0]


        num_points = root // 2 + 1
        next_label = num_points + 1

        node_list = self.bfs_from_hierarchy(hierarchy, root)
        relabel = np.empty(root + 1, dtype=np.intp)

        relabel[root] = num_points
        result_list = []
        ignore = np.zeros(len(node_list), dtype=int)

        for node in node_list:
            if ignore[node] or node < num_points:
                continue

            children = hierarchy[node - num_points]
            left = int(children[0])
            right = int(children[1])

            if children[2] > 0.0:
                lambda_value = 1.0 / children[2]
            else:
                lambda_value = INFTY

            if left >= num_points:
                left_count = int(hierarchy[left - num_points][3])
            else:
                left_count = 1

            if right >= num_points:
                right_count = int(hierarchy[right - num_points][3])
            else:
                right_count = 1

            if left_count >= self.mpts and right_count >= self.mpts:
                relabel[left] = next_label
                next_label += 1
                result_list.append((relabel[node], relabel[left], lambda_value, left_count))

                relabel[right] = next_label
                next_label += 1
                result_list.append((relabel[node], relabel[right], lambda_value, right_count))

            elif left_count < self.mpts and right_count < self.mpts:
                for sub_node in self.bfs_from_hierarchy(hierarchy, left):
                    if sub_node < num_points:
                        result_list.append((relabel[node], sub_node, lambda_value, 1))
                    ignore[sub_node] = True

                for sub_node in self.bfs_from_hierarchy(hierarchy, right):
                    if sub_node < num_points:
                        result_list.append((relabel[node], sub_node, lambda_value, 1))
                    ignore[sub_node] = True

            elif left_count < self.mpts:
                relabel[right] = relabel[node]
                for sub_node in self.bfs_from_hierarchy(hierarchy, left):
                    if sub_node < num_points:
                        result_list.append((relabel[node], sub_node, lambda_value, 1))
                    ignore[sub_node] = True

            else:
                relabel[left] = relabel[node]
                for sub_node in self.bfs_from_hierarchy(hierarchy, right):
                    if sub_node < num_points:
                        result_list.append((relabel[node], sub_node, lambda_value, 1))
                    ignore[sub_node] = True

        return np.array(result_list, dtype=[('parent', np.intp),
                                            ('child', np.intp),
                                            ('lambda_val', float),
                                            ('child_size', np.intp)])


    def compute_stability(self):
        largest_child = max(self.condensed_tree['child'].max(), self.condensed_tree['parent'].min())
        num_clusters = self.condensed_tree['parent'].max() - self.condensed_tree['parent'].min() + 1
        smallest_cluster = self.condensed_tree['parent'].min()

        sorted_child_data = np.sort(self.condensed_tree[['child', 'lambda_val']], axis=0)
        births_arr = np.nan * np.ones(largest_child + 1, dtype=np.double)

        sorted_children = sorted_child_data['child'].copy()
        sorted_lambdas = sorted_child_data['lambda_val'].copy()

        parents = self.condensed_tree['parent']
        sizes = self.condensed_tree['child_size']
        lambdas = self.condensed_tree['lambda_val']

        current_child = -1
        min_lambda = 0

        for row in range(sorted_child_data.shape[0]):
            child = int(sorted_children[row])
            lambda_ = sorted_lambdas[row]

            if child == current_child:
                min_lambda = min(min_lambda, lambda_)
            elif current_child != -1:
                births_arr[current_child] = min_lambda
                current_child = child
                min_lambda = lambda_
            else:
                # Initialize
                current_child = child
                min_lambda = lambda_

        if current_child != -1:
            births_arr[current_child] = min_lambda
        births_arr[smallest_cluster] = 0.0

        result_arr = np.zeros(num_clusters, dtype=np.double)

        for i in range(self.condensed_tree.shape[0]):
            parent = parents[i]
            lambda_ = lambdas[i]
            child_size = sizes[i]
            result_index = parent - smallest_cluster

            result_arr[result_index] += (lambda_ - births_arr[parent]) * child_size

        result_pre_dict = np.vstack((np.arange(smallest_cluster, self.condensed_tree['parent'].max() + 1),
                                    result_arr)).T

        return dict(result_pre_dict)



    def bfs_from_cluster_tree(self,bfs_root):
        result = []
        to_process = np.array([bfs_root], dtype=np.intp)

        while to_process.shape[0] > 0:
            result.extend(to_process.tolist())
            to_process = self.condensed_tree['child'][np.in1d(self.condensed_tree['parent'], to_process)]

        return result

    def do_labelling(self,clusters, cluster_label_map, allow_single_cluster, cluster_selection_epsilon, match_reference_implementation):

        root_cluster = min(self.condensed_tree['parent'])

        result_arr = np.empty(root_cluster, dtype=np.intp)

        result = result_arr.view(np.intp)

        child_array = self.condensed_tree['child']
        parent_array = self.condensed_tree['parent']
        lambda_array = self.condensed_tree['lambda_val']

        union_find = TreeUnionFind(max(parent_array) + 1)

        for n in range(self.condensed_tree.shape[0]):
            child = child_array[n]
            parent = parent_array[n]
            if child not in clusters:
                union_find.union(parent, child)

        for n in range(root_cluster):
            cluster = union_find.find(n)
            if cluster < root_cluster:
                result[n] = -1
            elif cluster == root_cluster:
                result[n] = -1
            else:
                if match_reference_implementation:
                    point_lambda = lambda_array[child_array == n][0]
                    cluster_lambda = lambda_array[child_array == cluster][0]

                    if point_lambda > cluster_lambda:
                        result[n] = cluster_label_map[cluster]
                    else:
                        result[n] = -1

        return result_arr

    def get_clusters(self,stability, cluster_selection_method='eom',
                    allow_single_cluster=False,
                    match_reference_implementation=False,
                    cluster_selection_epsilon=0.0,
                    max_cluster_size=0):
        

        node_list = sorted(stability.keys(), reverse=True)[:-1]

        cluster_tree = self.condensed_tree[self.condensed_tree['child_size'] > 1]

        is_cluster = {cluster: True for cluster in node_list}

        num_points = np.max(self.condensed_tree[self.condensed_tree['child_size'] == 1]['child']) + 1

        max_lambda = np.max(self.condensed_tree['lambda_val'])

        if max_cluster_size <= 0:
            max_cluster_size = num_points + 1

        cluster_sizes = {child: child_size for child, child_size
                        in zip(cluster_tree['child'], cluster_tree['child_size'])}



        if cluster_selection_method == 'eom':
            for node in node_list:
                child_selection = (cluster_tree['parent'] == node)
                subtree_stability = np.sum([
                    stability[child] for
                    child in cluster_tree['child'][child_selection]])
                
                if subtree_stability > stability[node] or cluster_sizes[node] > max_cluster_size:
                    is_cluster[node] = False
                    stability[node] = subtree_stability
                else:
                    for sub_node in self.bfs_from_cluster_tree(node):
                        if sub_node != node:
                            is_cluster[sub_node] = False

        clusters = set([c for c in is_cluster if is_cluster[c]])
        cluster_map = {c: n for n, c in enumerate(sorted(list(clusters)))}
        reverse_cluster_map = {n: c for c, n in cluster_map.items()}


        labels = self.do_labelling(clusters, cluster_map,
                            allow_single_cluster, cluster_selection_epsilon,
                            match_reference_implementation)
        


        return labels

    def fit(self):
        
        # Queremos salvar, as arestas da MST, Árvore condensada e os LABELS


        # Salva as arestas da MST
        self.mst_edges = self.mst_edges[np.argsort(self.mst_edges[:,2]),:]

        # Não precisamos guardar
        self.linkage = self.label()

        # Salva a árvore condensada
        self.condensed_tree = self.condense_tree(self.linkage)

        # Não precisamos salvar
        stability_dict = self.compute_stability()


        cluster_selection_method="eom"
        allow_single_cluster=False
        match_reference_implementation=True
        cluster_selection_epsilon=0.0
        max_cluster_size=0

        # Salva os labels
        self.labels = self.get_clusters(
                stability_dict,
                cluster_selection_method,
                allow_single_cluster,
                match_reference_implementation,
                cluster_selection_epsilon,
                max_cluster_size,
            )

        return


    def fit_linkage(self,app_hdbscan):
        max_node = 2 * app_hdbscan.linkage.shape[0]
        num_points = max_node - (app_hdbscan.linkage.shape[0] - 1)

        parent_array = np.arange(num_points, max_node + 1)
        app_linkage = pd.DataFrame({'parent': parent_array,
                                'left_child': app_hdbscan.linkage[:,0],
                                'right_child': app_hdbscan.linkage[:,1],
                                'distance': app_hdbscan.linkage[:,2],
                                'size': app_hdbscan.linkage[:,3]})[['parent', 'left_child', 'right_child', 'distance', 'size']]
        return app_linkage
