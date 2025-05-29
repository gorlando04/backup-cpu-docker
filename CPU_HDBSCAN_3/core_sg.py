import numpy as np
import pynndescent
import networkx as nx
import time
import pandas as pd


################################################################################################################
#                                                                                                              #
#                                           HUBNESS                                                            #     
#                                                                                                              #         
#                                                                                                              #     
#                                                                                                              #     
#                                                                                                              # 
################################################################################################################

def merge_sort_dict_by_values(dictionary):

    if len(dictionary) <= 1:
        return dictionary

    middle = len(dictionary) // 2
    left_half = merge_sort_dict_by_values({k: dictionary[k] for k in list(dictionary.keys())[:middle]})
    right_half = merge_sort_dict_by_values({k: dictionary[k] for k in list(dictionary.keys())[middle:]})

    return merge_dicts_by_values(left_half, right_half)



def merge_dicts_by_values(dict1, dict2):


    result = {}
    dict1_keys = list(dict1.keys())
    dict2_keys = list(dict2.keys())
    i, j = 0, 0

    while i < len(dict1_keys) and j < len(dict2_keys):
        if dict1[dict1_keys[i]] <= dict2[dict2_keys[j]]:
            result[dict1_keys[i]] = dict1[dict1_keys[i]]
            i += 1
        else:
            result[dict2_keys[j]] = dict2[dict2_keys[j]]
            j += 1

    while i < len(dict1_keys):
        result[dict1_keys[i]] = dict1[dict1_keys[i]]
        i += 1

    while j < len(dict2_keys):
        result[dict2_keys[j]] = dict2[dict2_keys[j]]
        j += 1

    return result

import math

def euclidean_distance(point1, point2):

    
    squared_distance = sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2))
    distance = math.sqrt(squared_distance)
    
    return distance


class S_CoreSG:

    def __init__(self,data,k) -> None:
        self.data = data
        self.N,self.dim = data.shape
        self.k = k

    def build_kNNG(self):

        index = pynndescent.NNDescent(self.data,metric='euclidean',n_neighbors=self.k+1,random_state=int(time.time()))
        index.prepare()


        self.indices,self.D = index.query(self.data,k=self.k+1)

        self.indices = self.indices[:,1:]
        self.D = self.D[:,1:]
        self.grafo = nx.Graph()

        ## Conect the kNNG nodes to check if it is connected
        self.eucli_distance_dict = {}

        for index,i in enumerate(self.indices):

            # I = vizinhos, index = Valores crscentes de 0 à N-1
            
            core_distance_i = self.D[index][-1]

            lista = []
            for index2,j in enumerate(i):

                # J = index de um vizinho individual index2 = posição do vizinho.
                core_distance_j = self.D[j][-1]


                weight = core_distance_i

                if core_distance_i < core_distance_j:
                    weight = core_distance_j
                
                # Distância euclidiana entre os pontos está no vetor de distâncias
                eucli = self.D[index][index2]

                if weight < eucli:
                    weight = eucli

                
                if index in self.indices[j]:
                    if index < j:
                        # Adicionar
                        self.grafo.add_weighted_edges_from([(index,j,weight)])

                        if j not in self.eucli_distance_dict:
                            self.eucli_distance_dict[j] = {}
                        
                        self.eucli_distance_dict[j][index] = eucli

                else:
                    # Adicionar também
                    maior = index
                    menor = j
                    if menor > maior:
                        aux = maior
                        maior = menor
                        menor = aux
                    self.grafo.add_weighted_edges_from([(menor,maior,weight)])

                    if maior not in self.eucli_distance_dict:
                        self.eucli_distance_dict[maior] = {}
                
                    self.eucli_distance_dict[maior][menor] = eucli

        self.knng = self.grafo.copy()

        return

    def is_connected(self):

        return nx.is_connected(self.grafo)

    def limite(self):
        import math

        j = int(math.sqrt(self.N))
    
    
        return j

    def get_num_hubs(self):

        if hasattr(self, "hubs"):
            return len(self.hubs)
        return 0

    def get_hubs(self):


        self.graus = {}

        for index,i in enumerate(self.indices):
    
    
            if index not in self.graus:
                self.graus[index] = 0
        
        
            for neig in i:
        
                if int(neig) not in self.graus:
                    self.graus[int(neig)] = 1
                else:
                    self.graus[int(neig)] += 1
                


        
        hubs = int(self.limite())

        self.graus = merge_sort_dict_by_values(self.graus)

        self.hubs = list(self.graus.keys())[:hubs]

        minus = self.untie()

        # Aqui tem apenas vértices com graus menores que o empate.
        self.hubs = self.hubs[:minus]

        self.scores =  merge_sort_dict_by_values(self.scores)
        
        self.sorted_scores = list(self.scores.keys())


        aux = self.hubs.copy()

        self.hubs = np.zeros(self.limite(),dtype=np.int64)

        
        for i in range(self.limite()):

            if i < minus:
                self.hubs[i] = aux[i]
            else:
                self.hubs[i] = self.sorted_scores[i-minus]

        
    def connect_new_nodes(self):
        

        self.hubs = sorted(list(self.hubs))

        last = int(self.hubs[-1])
        if last not in self.eucli_distance_dict:
            self.eucli_distance_dict[last] = {}
        
        for index,i in enumerate(self.hubs[:-1]):
            
            # Index = posição no array de indexes, i = valor do índice de anti-hub

            core_distance_i = self.D[last][-1]

            core_distance_j = self.D[i][-1]

            weight = core_distance_i

            if core_distance_i < core_distance_j:
                weight = core_distance_j
            

            eucli = euclidean_distance(self.data[i],self.data[last])

        
            self.eucli_distance_dict[last][i] = eucli

            if weight < eucli:
                weight = eucli


            self.grafo.add_weighted_edges_from([(i,last,weight)])
    
    def verify_ties(self):

        treshold = self.graus[self.hubs[-1]]

        soma = 0
        soma2 = 0

        for i in self.graus:

            if treshold == self.graus[i]:
                soma += 1
            elif treshold > self.graus[i]:
                soma2 += 1

        return (soma - (self.limite() - soma2),soma)



    def verify_ties2(self):

        treshold = self.scores[self.hubs[-1]]


        soma = 0

        for i in self.scores:

            if treshold == self.scores[i]:
                soma += 1


        return soma


    def untie(self):

        treshold = self.graus[self.hubs[-1]]

        self.scores = {}

        soma = 0
        for i in self.graus:

            if treshold == self.graus[i]:
                score = 0
                for k in self.indices[i]:
                    score += self.graus[k]
                self.scores[i] = score
                
            elif treshold > self.graus[i]:
                soma += 1

        return soma

    def build_core_sg(self):



        self.build_kNNG()


        # Encontra os índices dos anti-hubs
        self.get_hubs()

        # Conecta os S_CoreSGs entre si
        self.connect_new_nodes()

        # Verifica se o novo kNNG é conexo
        conexo = self.is_connected()

        if not conexo:
            print("O Novo kNNG nao eh conexo\n")
            exit(0)

        # Extrai a MST do grafo usando o algoritmo de PRIM
        MST = nx.minimum_spanning_tree(self.grafo,algorithm='prim')


        self.core_sg = nx.compose(self.knng,MST)

        return
    

    def get_core_distance(self,mpts):

        return self.D[:,mpts-1]

    def update_coreSG(self,mpts):

        core_distance = self.get_core_distance(mpts)

        new_core_sg = self.core_sg.copy()

        for i in new_core_sg:
            
            l = [i,-1]
            for j in new_core_sg[i]:

                
                l[1] = j
                new_mrd = [core_distance[i],core_distance[j], self.eucli_distance_dict[max(l)][min(l)]]

                new_core_sg[i][j]['weight'] = max(new_mrd)

        return new_core_sg


    def get_core_sg(self):
        return self.core_sg.copy()

