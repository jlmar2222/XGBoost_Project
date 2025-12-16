# Aqui vanmos a empezar a codificar el XGBoost

import numpy as np


class Stump:
    def __init__(self, data,k):
        super().__init__()       

        self.data = data # data contiene y, X, g, h, w en ese orden        
        self.y = self.data[:,0:1]
        self.X = self.data[:,1:k]
        self.g = self.data[:,k:k+1]
        self.h = self.data[:,k+1:k+2]  

        self.w = self.data[:,k+2:k+3]

        # Guardará aqui los datos una vez ejecutado el split
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
    

    def do_split(self,eps,alpha):
        # revisar porque alomejor epsilon no deberia de ser estático y depende de w(D) como
        # pone en el apendice del Weighted Quantile Sketch 

        # basicamente se mide que particion (split) genera mas Gain 
        # basandonos en esa ecuaion y la que más valor de, es la  que se utiliza
        # recuerda que aunque los weights (hessianos) sean compartidos por filas
        # el hecho de que el sort sea por columna (feature) asegura o permite
        # que los quantiles sean feature dependent

        D = np.concatenate((self.X, self.h), axis=1) # Espacio de instancias con pesos
        splits = {} # mejor split por feature
        
        for k in range(D.shape[1]-1): # quitamos la dimension de h y recorremos solo los features
        #(1) Get Candidates
            candidates = [] # Serán los Upper Bounds de los bins representativos de los quantiles

            #(i) Sort/Filtrar de menor a mayor valores en la columna            
            idx = np.argsort(D[:, k]) # obtenemos indices ordenados
            d_k = D[:,[k,-1]][idx] # generamos Columna organizada en base a esos indices

            #(ii) Marcamos Candidatos segun Weighted Quantile Criteria (weight=h) 
            # acumulamos datos hasta que la suma de sus weights se aprox. eps           
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
            father = self.data[:,[k+1,-3,-2]] # coge Columna del feature, gradiante y hessiano
            G = np.sum(father[:,-2])
            H = np.sum(father[:,-1])

            Father_Gain = (G**2)/(H + alpha)

            best_score = -np.inf

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

                if best_score < score:
                    best_score = score
                    split = candidates[i]
                    

                else:
                    best_score = best_score
                    split = split

            splits[k+1] = (best_score,split)

        # devulve el valor del diccionario con maximo score
        best_split = max(splits.items(), key = lambda x: x[1][0])  # ('feature' : (score,split_value))

        feature = best_split[0] # feature splited
        threshold = best_split[1][1] # threshold selected

        # (3) Do the Actual Split
        L_data = self.data[self.data[:,feature] < threshold]
        R_data = self.data[self.data[:,feature] >= threshold]

        # Create Stump instances based in this new splits
        # formamos Devolvemos 2 nuevos Stumps dentro del msimo Stump
        # Ponemos un control para Evitar crear nodos vacíos (= None)
        L_son = Stump(L_data, k) if L_data.shape[0] > 0 else None # k = self.X.shape[1]
        R_son = Stump(R_data, k) if R_data.shape[0] > 0 else None

        # Guardamos resultados par ael objeto
        self.feature = feature
        self.threshold = threshold
        self.left = L_son
        self.right = R_son

        return L_son, R_son



class Tree():

    def __init__(self, data, k, max_depth):
        super().__init__()
        self.k = k
        self.data = data
        self.max_depth = max_depth 

        self.root = None
        self.leaves = None
        # revisar porque puede ser que arboles sean mas pequeños que maxdepth 
        # esto puede hacer que pete imagino 
        self.new_data = None
    
    def build_tree(self, data, eps, alpha, depth = 0, min_samples = 1):

        node = Stump(data,self.k)

        if depth >= self.max_depth or data.shape[0] < min_samples:        
            return node

        # Solo continuamos construyendo si el nodo no es None
        L_node, R_node = node.do_split(eps,alpha) 
        if L_node is not None:
            node.left = self.build_tree(L_node.data,eps,alpha, depth = depth + 1)
        if R_node is not None:
            node.right = self.build_tree(R_node.data, eps, alpha, depth = depth + 1)

        return node

    def fit(self, data, eps, alpha): # desarrolla el arbol en su plenitud, desde la raiz
        self.root = self.build_tree(data, eps, alpha)


    def get_leaves(self,node): # recorre el arbol y devuelve los nodos hoja
        
    # Caso base: si el nodo no tiene hijos, es hoja
        if node.left is None and node.right is None:
            return [node]  # devolvemos la data del nodo hoja
        
        leaves = []
        if node.left:
            leaves += self.get_leaves(node.left)
        if node.right:
            leaves += self.get_leaves(node.right)        
        
        self.leaves = leaves
        return leaves      
        
        #return leafs = [leaf,leaf,...,leaf] # lista de nodos hoja (Stumps)


    def do_predicttion(self,leaves,alpha): # prectition = weight of the leaf based on equation (5)
        new_leafs = []
        for leaf in leaves:

            G_j = np.sum(leaf.data[:,-2])
            H_j = np.sum(leaf.data[:,-1])

            weight = - (G_j / (H_j + alpha))

            leaf.data[:,-1] = weight # ponemos las nuevas predicciones en la columna de w
            new_leafs.append(leaf.data)
        
        self.new_data = np.vstack(new_leafs)    

           
    
                      

###   PRUEBA   ########################    

np.random.seed(42)  # <- seed fija

# Definimos n y k para el ejemplo
n = 20  # Número de filas
k = 2  # Número de columnas para X

# Creamos las matrices individuales con dimensiones aleatorias
y = np.random.rand(n, 1) # (n, 1)
X = np.random.rand(n, k) # (n, k)
g = np.random.rand(n, 1) # (n, 1)
h = np.random.rand(n, 1) # (n, 1)

w = np.random.rand(n, 1) # (n, 1) añadimo suna ultima fila para los pesos

# Concatenamos las matrices horizontalmente para formar 'data'
data = np.concatenate((y, X, g, h, w), axis=1)


model = Tree(data,3,3)

model.fit(data,0.05,0.05)
model.get_leaves(model.root)

model.do_predicttion(model.leaves,0.05)

new_data = model.new_data


#print(model.leaves[1].data)
#print(new_data)

#print(new_data)
print(data)