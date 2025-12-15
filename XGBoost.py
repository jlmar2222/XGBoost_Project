# Aqui vanmos a empezar a codificar el XGBoost

import numpy as np


class Stump:
    def __init__(self, data,k):
        super().__init__()
        self.data = data 
        self.y = self.data[:,0:1]
        self.X = self.data[:,1:k]
        self.g = self.data[:,k:k+1]
        self.h = self.data[:,k+1:k+2]  
    
        pass

    def do_split(self,eps,alpha):
        # revisar porque alomejor epsilon no deberia de ser estático y depende de w(D) como
        # pone en el apendice del Weighted Quantile Sketch 

        D = np.concatenate((self.X, self.h), axis=1) # Espacio de instancias con pesos
        splits = {} # mejor split por feature
        
        for k in range(D.shape[1]):
        #(1) Get Candidates
            candidates = [] # Serán los Upper Bounds de los bins representativos de los quantiles

            #(i) Sort/Filtrar de menor a mayor valores en la columna            
            idx = np.argsort(D[:, k]) # obtenemos indices ordenados
            d_k = D[:,[k,-1]][idx] # generamos Columna organizada en base a esos indices

            #(ii) Marcamos Candidatos segun Weighted Quantile Criteria (weight=h)            
            accumulated = 0
            for n in range(D.shape[0]):
                
                h = d_k[n,-1] # weight = hessian
                x_i = d_k[n,0] # instance del feature

                if accumulated + h < eps:
                    accumulated += h
                else:
                    candidates.append(x_i) # candidates ~ [min(x_i), max(x_i)]   
                    accumulated = 0             
        #(2) Save Scores (based on Gains => Equación (7) del paper original)
            father = self.data[:,[k+1,-2,-1]] # coge Columna del feature, gradiante y hessiano
            G = np.sum(father[:,-2])
            H = np.sum(father[:,-1])

            Father_Gain = (G**2)/(H + alpha)

            best_candidate = -np.inf

            for i in range(len(candidates)):

                left_son = father[father[:, 0] < candidates[i]]
                # Calculamos gradiante/hessiano acumulado del posible hijo (a la izquierda)
                G_l = np.sum(left_son[:,-2])
                H_l = np.sum(left_son[:,-1])
                # gradiante/hessiano acumulado del hijo derecho es la diferencia entre el izquierdo y el padre
                G_r = G - G_l
                H_r = H - H_l 

                Left_Son_Gain = (G_l**2)/(H_l + alpha)
                Right_Son_Gain= (G_r**2)/(H_r + alpha)

                score = Left_Son_Gain + Right_Son_Gain - Father_Gain # based on Equation (7)

                if best_candidate < score:
                    best_candidate = score
                    split = candidates[i]
                    

                else:
                    best_candidate = best_candidate
                    split = split

            splits[k+1] = split

        # devulve el valor del diccionario maximo
        best_split = max(splits.items(), key = lambda x: x[1])  # ('feature' : split_value)

    # basicamente se mide que particion (split) genera mas Gain 
    # basandonos en esa ecuaion y la que más valor de, es la  que se utiliza
    # recuerda que aunque los weights (hessianos) sean compartidos por filas
    # el hecho de que el sort sea por columna (feature) asegura o permite
    # que los quantiles sean feature dependent

        # yo haria que esta funcion me devolviera 2 hijos  de un nodo padre es decir
        # es decir que me filtre una base de datos dada en 2 partes basandose en el split optimo
        # calculado 



