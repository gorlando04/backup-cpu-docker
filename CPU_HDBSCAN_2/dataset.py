import numpy as np
import sklearn.datasets as data


import os

import pandas as pd


################################################################################################################
#                                                                                                              #
#                                           MULTI-DATASET                                                      #     
#                                                                                                              #         
#                                                                                                              #     
#                                                                                                              #     
#                                                                                                              # 
################################################################################################################



################################################################################################################
#                                                                                                              #
#                                           DATASETS CREATION                                                  #     
#                                                                                                              #         
#                                                                                                              #     
#                                                                                                              #     
#                                                                                                              # 
################################################################################################################

## Join the arrays that have differente probabilistic distributions
def join_sample(data):
    
    
    sample = data[0]
    for i in range(1,len(data)):
        sample = np.concatenate((sample,data[i]))
    
    return sample



## Create a dataset with bicluster distributions
def biclust_dataset(N,dim):
    #Building make_bicluster dataset
    from sklearn.datasets import make_biclusters
    X0, rows,_ = make_biclusters(
    shape=(N, dim), n_clusters=2, noise=.4,minval=-12,maxval=10, shuffle=False)
    
    return X0

## Create dataset with make_blobs distribution
def blobs_dataset(N,dim,n_clust=5,cluster_std=.6):
    #Building make_blobs dataset
    from sklearn.datasets import make_blobs
    X1, _ = make_blobs(n_samples=N, centers=n_clust, n_features=dim,
                   cluster_std=cluster_std)
    return X1

## Get the datasets with the propreties that is especified, and call the make col func
def get_artifical_db(N,dim,n_clust=5,cluster_std=.6):
    
    old_n = N
    N = N//2
    
    x0 = biclust_dataset(N,dim)
    
    x1 = blobs_dataset(N,dim,n_clust,cluster_std)
    
    data = [x0,x1]
    
    sample = join_sample(data)

    
    np.random.shuffle(sample)
    return sample

## Create the dataset by calling the functions above and check their integrity
def create_dataset(N,dim,n_clust=5,cluster_std=.6):
    
    sample = get_artifical_db(N,dim,n_clust,cluster_std)

      
    return sample.astype(np.float32)



def sparse_dataset(N,D,center=25,std=2):
        from sklearn.datasets import make_blobs


        # Parameters for the dense clusters
        dense_cluster_1_center = (-center,) * D
        dense_cluster_2_center = (center,) * D
        num_samples_dense = int(N * .3333)

        # Parameters for the sparse cluster
        sparse_cluster_center = (0,) * D
        num_samples_sparse = int(N * .33341)

        # Create data for the dense clusters
        dense_data_1, _ = make_blobs(n_samples=num_samples_dense,n_features=D, centers=[dense_cluster_1_center], cluster_std=std)
        dense_data_2, _ = make_blobs(n_samples=num_samples_dense, n_features=D, centers=[dense_cluster_2_center], cluster_std=std)

        # Create data for the sparse cluster
        sparse_data, _ = make_blobs(n_samples=num_samples_sparse, n_features=D, centers=[sparse_cluster_center], cluster_std=10.0)

        # Combine the datasets
        final_dataset = np.vstack((dense_data_1, dense_data_2, sparse_data))

        np.random.shuffle(final_dataset)

        return final_dataset


################################################################################################################
#                                                                                                              #
#                                           MOONS                                                              #     
#                                                                                                              #         
#                                                                                                              #     
#                                                                                                              #     
#                                                                                                              # 
################################################################################################################


def moons_db(N):
    moons, _ = data.make_moons(n_samples=N//2, noise=0.05)
    blobs, _ = data.make_blobs(n_samples=N//2, centers=[(-0.75,2.25), (1.0, 2.0)], cluster_std=0.25)
    test_data = np.vstack([moons, blobs])

    return test_data.astype(np.float32)




class Dataset: 

    def __init__(self,distributions,N,dim,shuffle_) -> None:
        
        self.distributions = []
        self.N = N
        self.dim = dim
        self.shuffle_ = shuffle_
        self.name = distributions[0]['name'] 
 
        for i in distributions:
            self.distributions.append(i)
        
    def create_partial_dataset(self):

        self.partial_data = []

        for idx,distr in enumerate(self.distributions):

            sample_size = int(distr['weight'] * self.N)

            aux = distr["type"](size=(sample_size,self.dim), **distr["kwargs"])

            self.partial_data.append(aux)

        return 

    def concatenate_data(self):
        sample = self.partial_data[0]

        for i in range(1,len(self.partial_data)):
            sample = np.concatenate((sample,self.partial_data[i]))

        self.data = sample

        del sample
        return

    def get_dataset(self):
        return self.data

    def shuffle_ds(self):
        np.random.shuffle(self.data)



################################################################################################################
#                                                                                                              #
#                                           DISTRIBUTIONS                                                      #     
#                                                                                                              #         
#                                                                                                              #     
#                                                                                                              #     
#                                                                                                              # 
################################################################################################################

class Distributions:

    def __init__(self) -> None:
        
        pass


    def get_beta():

        beta_distributions = [
        {"name":'beta' ,"type": np.random.beta, "kwargs": {"a":2 , "b":40},"weight":.3333},
            {"name":'beta' ,"type": np.random.beta, "kwargs": {"a":40 , "b":2},"weight":.33341},
            {"name":'beta' ,"type": np.random.beta, "kwargs": {"a":20 , "b":20},"weight":.3333}
        ]

        return beta_distributions
    
    def get_chi():
        chi_distributions = [
   {"name":'chi' ,"type": np.random.chisquare, "kwargs": {"df":4},"weight":.3333},
    {"name":'chi' ,"type": np.random.chisquare, "kwargs": {"df":52},"weight":.33341},
    {"name":'chi' ,"type": np.random.chisquare, "kwargs": {"df":140},"weight":.3333},
] 
        return chi_distributions
    
    def get_gamma():
        gamma_distributions = [
   {"name":'gamma' ,"type": np.random.gamma, "kwargs": {"shape":10, "scale":8.5},"weight":.3333},
    {"name":'gamma' ,"type": np.random.gamma, "kwargs": {"shape":40, "scale":9},"weight":.33341},
    {"name":'gamma' ,"type": np.random.gamma, "kwargs": {"shape":90, "scale":9.5},"weight":.3333}
    
] 
        return gamma_distributions
    
    def get_gumbel():
        gumbel_distributions = [
   {"name":'gumbel' ,"type": np.random.gumbel, "kwargs": {"loc":0, "scale":5},"weight":.3333},
    {"name":'gumbel' ,"type": np.random.gumbel, "kwargs": {"loc":85, "scale":7},"weight":.33341},
    {"name":'gumbel' ,"type": np.random.gumbel, "kwargs": {"loc":170, "scale":10},"weight":.3333},
    
] 
        return gumbel_distributions
    
    def get_laplace():
        laplace_distributions = [
   {"name":'laplace' ,"type": np.random.laplace, "kwargs": {"loc":0, "scale":5},"weight":.3333},
    {"name":'laplace' ,"type": np.random.laplace, "kwargs": {"loc":95, "scale":7},"weight":.33341},
    {"name":'laplace' ,"type": np.random.laplace, "kwargs": {"loc":210, "scale":10},"weight":.3333},
    
] 
        return laplace_distributions
    
    def get_logistic():
        logistic_distributions = [
   {"name":'logistic' ,"type": np.random.logistic, "kwargs": {"loc":0, "scale":5},"weight":.3333},
    {"name":'logistic' ,"type": np.random.logistic, "kwargs": {"loc":95, "scale":7},"weight":.33341},
    {"name":'logistic' ,"type": np.random.logistic, "kwargs": {"loc":210, "scale":10},"weight":.3333},
    
] 
        return logistic_distributions
    

    def get_poison():
        poisson_distributions = [
  {"name":'poisson' ,"type": np.random.poisson, "kwargs": {"lam":10},"weight":.3333},
   {"name":'poisson' ,"type": np.random.poisson, "kwargs": {"lam":50},"weight":.33341},
    {"name":'poisson' ,"type": np.random.poisson, "kwargs": {"lam":100},"weight":.3333}
    
]
        return poisson_distributions
    
    def get_uniform():

        uniform_distributions = [
   {"name":'uniform' ,"type": np.random.uniform, "kwargs": {"low":1, "high":1.5},"weight":.3333},
    {"name":'uniform' ,"type": np.random.uniform, "kwargs": {"low":1.6, "high":2.1},"weight":.33341},
    {"name":'uniform' ,"type": np.random.uniform, "kwargs": {"low":2.2, "high":2.7},"weight":.3333}

    
] 
        return uniform_distributions

    
    def get_vonmisses():
        vonmises_distributions = [
   {"name":'vonmises' ,"type": np.random.vonmises, "kwargs": {"mu":1, "kappa":10},"weight":.3333},
    {"name":'vonmises' ,"type": np.random.vonmises, "kwargs": {"mu":4, "kappa":17},"weight":.33341},
    {"name":'vonmises' ,"type": np.random.vonmises, "kwargs": {"mu":5.6, "kappa":20},"weight":.3333}

    
] 
        return vonmises_distributions
    

def get_20news(d=1000):

    from sklearn.datasets import fetch_20newsgroups

    from sklearn.feature_extraction.text import TfidfVectorizer


    vectorizer = TfidfVectorizer()

    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

    tfidf_matrix = vectorizer.fit_transform(newsgroups_train.data)


    # Get the feature names (terms) from the vectorizer
    feature_names = vectorizer.get_feature_names_out()

    # Sum the TF-IDF values for each term across all documents
    tfidf_scores = tfidf_matrix.sum(axis=0)

    # Sort the terms based on their TF-IDF scores
    sorted_indices = np.array(tfidf_scores.argsort()[0, ::-1])

    # Select the top 1000 terms with the highest TF-IDF values
    top_terms_indices = sorted_indices[0,:d]
    # Reshape the dataset using the top terms
    reshaped_matrix = tfidf_matrix[:, top_terms_indices]

    # Convert the reshaped matrix to a dense array
    reshaped_array = reshaped_matrix.toarray()

    del reshaped_matrix, top_terms_indices, sorted_indices, tfidf_matrix, tfidf_scores, vectorizer, newsgroups_train

    np.random.shuffle(reshaped_array)

    return reshaped_array
