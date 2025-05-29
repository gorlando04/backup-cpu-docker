from dataset import Dataset,Distributions,create_dataset,moons_db,get_20news,sparse_dataset

from core_sg import S_CoreSG

from app_hdbscan import Approximate_HDBSCAN

from eval import HAI,Evaluate

import numpy as np
import pandas as pd
import networkx as nx
import hdbscan
import sys
import time


np.random.seed(seed=int(time.time())) 

# Valor de K
K = 50
Ks = [50,32,20,10]


D = 2
N = int(1e4)
dataset = 'standart'

args = sys.argv[1:]



while args:
    a = args.pop(0)
    if a == '-name':      dataset = args.pop(0)
    elif a == '-D':       D = int(args.pop(0)) 
    elif  a == '-N':      N = int(args.pop(0)) 
    elif a == '-iter':    id = args.pop(0)
    elif a == '-kmax':    K = int(args.pop(0))
    #elif a == '-kmin':    K_min = int(args.pop(0))
    #elif a == '-ksteps':  K_steps = int(args.pop(0)) 
    else:
        print("argument %s unknown" % a, file=sys.stderr)
        sys.exit(1)

# Tratar caso dos datasets reais


print(f"Argumentos: N = {N} Dim = {D} Dataset_Name = {dataset} e MPTS = {K} ID = {id}")
args = {'N':N,'Dim':D,'name':dataset}




################################################################################################################
#                                                                                                              #
#                                           INSTANTIATE DATASET                                                #     
#                                                                                                              #         
#                                                                                                              #     
#                                                                                                              #     
#                                                                                                              # 
################################################################################################################


def instatiate_dataset(N,D,name):
    

    if name == 'gaussian-7-noOverlap': 
        return create_dataset(N,D)
    
    elif name == 'gaussian-5-noOverlap': 
        return create_dataset(N,D,n_clust=3)

    if name == 'gaussian-7-Overlap': 
        return create_dataset(N,D,n_clust=5,cluster_std=2.5)

    if name == 'gaussian-5-Overlap': 
        return create_dataset(N,D,n_clust=3,cluster_std=2.5)
     
    
    elif name == 'gaussian-sparse_noOverlap':
        return sparse_dataset(N,D)
    elif name == 'gaussian-sparse-Overlap':
        return sparse_dataset(N,D,center=18,std=1)
    
    elif name == 'beans':

        # Requirement = pip install ucimlrepo
        from ucimlrepo import fetch_ucirepo 
        
        dry_bean_dataset = fetch_ucirepo(id=602) 

        # data (as pandas dataframes) 
        X = dry_bean_dataset.data.features 

        del dry_bean_dataset

        X = X.to_numpy()

        # Normalizar standart
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X  = scaler.fit_transform(X)

        np.random.shuffle(X)

        return X
    
    elif name == '20news_300':

        return get_20news(d=300)


    elif name == '20news_500':

        return get_20news(d=500)


    
    dist = {}

    if name == 'beta':
        dist = Distributions.get_beta()

    elif name == 'chi':
        dist = Distributions.get_chi()

    elif name == 'gamma':
        dist = Distributions.get_gamma()

    elif name == 'gumbel':
        dist = Distributions.get_gumbel()


    elif name == 'laplace':
        dist = Distributions.get_laplace()


    elif name == 'logistic':
        dist = Distributions.get_logistic()


    elif name == 'poisson':
        dist = Distributions.get_poison()


    elif name == 'uniform':
        dist = Distributions.get_uniform()


    elif name == 'vonmisses':
        dist = Distributions.get_vonmisses()

    else:
        print("Nenhum dataset encontrado.")
        exit()

    ds = Dataset(dist,N,D,False) 

    ds.create_partial_dataset()

    ds.concatenate_data()

    ds.data = np.unique(ds.get_dataset(),axis=0)
    ds.N = ds.get_dataset().shape[0]

    ds.shuffle_ds()


    return ds.get_dataset()



################################################################################################################
#                                                                                                              #
#                                           EXACT HDBSCAN                                                      #     
#                                                                                                              #         
#                                                                                                              #     
#                                                                                                              #     
#                                                                                                              # 
################################################################################################################

class ClusteringHDBSCAN:

    def __init__(self,args):
        self.min_cluster_size = args['min_clust']
        self.exact = args['exact']

        


    def cluster(self,data):
        
        if self.exact:
            self.clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, algorithm='generic', metric='euclidean' , approx_min_span_tree=False, 
                                             match_reference_implementation=True,cluster_selection_method="eom",
                                allow_single_cluster=False,
                                cluster_selection_epsilon=0.0,
                                max_cluster_size=0)
                                        
    
        self.clusterer.fit(data)

        self.labels = self.clusterer.labels_




################################################################################################################
#                                                                                                              #
#                                           ESCOPO PRINCIPAL                                                   #     
#                                                                                                              #         
#                                                                                                              #     
#                                                                                                              #     
#                                                                                                              # 
################################################################################################################

def write_df(df,index,info):

    for i in info.keys():
        if i != 'gpu_res' and i != 'data':
            df.loc[index,i] = info[i]

    return


def main(args):

    X = instatiate_dataset(int(args['N']),int(args['Dim']),args['name']).astype(np.float64)

    N = X.shape[0]


    df_gpu = None

    file_name = '32d_core_sg.csv'

    try:   
        df_gpu = pd.read_csv(file_name)
    except:
        df_gpu = pd.DataFrame()

    index_ = df_gpu.shape[0]
    info = {}


    info['Name'] = args['name']
    info['N_sample'] = X.shape[0]
    info['Dim'] = X.shape[1]
    info['Size (GB)'] =  X.nbytes / 1e9
    info['id'] = id

    core_sg = S_CoreSG(X,K)
    core_sg.build_core_sg()

    del core_sg

    core_sg = S_CoreSG(X,K)


    for index,mpts in enumerate(Ks):

        # Tempo de uma iteração do Core-SG
        t0 = time.time()
        grafo = None
        if mpts == K:
            core_sg.build_core_sg()
            grafo = core_sg.get_core_sg()
        else:
            grafo = core_sg.update_coreSG(mpts)

        mst = nx.minimum_spanning_tree(grafo,algorithm='prim')

        app_hdbscan = Approximate_HDBSCAN(mst,N,mpts)
        app_hdbscan.fit()

        tf = time.time() - t0
        info[f'mpts_Iter-{index}'] = mpts
        info[f'Time_Iter-{index}_coresg'] = tf
        

        app_linkage = app_hdbscan.fit_linkage(app_hdbscan)
        app_labels = app_hdbscan.labels

        # Libera espaço
        del app_hdbscan,mst

        params = {'min_clust':mpts,'exact':True}
        t0 = time.time()
        exact = ClusteringHDBSCAN(params)
        exact.cluster(X)
        tf = time.time() - t0

        info[f'Time_Iter-{index}_hdbscan'] = tf

        # Inicia a construção da hierarquia para o cálculo do HAI
        exact_labels = exact.labels
        exact_linkage = exact.clusterer.single_linkage_tree_.to_pandas()
    
        # Libera espaço
        del exact 
    


        # Cria objeto para a avaliação dos resultados
        eval = Evaluate()

        # Calcula o ARI
        ari = eval.ARI_val(exact_labels,app_labels)

        info[f'ARI_Iter-{index}'] = ari

        Hier_exact = HAI(exact_linkage,N )
        Hier_exact.build_hierarchy()

        Hier_app =  HAI(app_linkage, N)
        Hier_app.build_hierarchy()

        #Calcula o HAI
        hai = eval.HAI_val(Hier_exact,Hier_app)


        info[f'HAI_Iter-{index}'] = hai
        write_df(df_gpu,index_,info) 


        df_gpu.to_csv(file_name, index=False)


    return 




if __name__ == '__main__':
    main(args)
    exit()

