# Aqui vanmos a empezar a codificar el XGBoost

import numpy as np


class Stump:
    def __init__(self, data,k):
        super().__init__()       

        self.data = data # data contiene y, X, g, h, w en ese orden   

        
        self.row_id = self.data[:,0:1] # indx = 0
        self.y = self.data[:,1:2] # indx = 1
        self.X = self.data[:,2:k+2] # indx = 2,...,k+1
        self.g = self.data[:,k+2:k+3] # indx = -3
        self.h = self.data[:,k+3:k+4] # indx = -2
        self.w = self.data[:,k+4:k+5] # indx = -1
        

        
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
            father = self.data[:,[k+2,-3,-2]] # coge Columna del feature, gradiante y hessiano
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

    def __init__(self, data, eps, alpha, k, max_depth):
        super().__init__()
        self.k = k
        self.eps = eps
        self.alpha = alpha
        self.data = data        
        self.max_depth = max_depth 

        self.root = None
        self.leaves = None
        # revisar porque puede ser que arboles sean mas pequeños que maxdepth 
        # esto puede hacer que pete imagino 
        self.new_data = None
    
    def build_tree(self, data, depth = 0, min_samples = 1):

        node = Stump(data,self.k)

        if depth >= self.max_depth or data.shape[0] < min_samples:        
            return node

        # Solo continuamos construyendo si el nodo no es None
        L_node, R_node = node.do_split(self.eps,self.alpha) 
        if L_node is not None:
            node.left = self.build_tree(L_node.data, depth = depth + 1)
        if R_node is not None:
            node.right = self.build_tree(R_node.data, depth = depth + 1)

        return node

    def fit(self, data): # desarrolla el arbol en su plenitud, desde la raiz
        self.root = self.build_tree(data)


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


    def do_predicttion(self,leaves): # prectition = weight of the leaf based on equation (5)
        new_leafs = []
        for leaf in leaves:

            G_j = np.sum(leaf.data[:,-3])
            H_j = np.sum(leaf.data[:,-2])

            weight = - (G_j / (H_j + self.alpha)) # based on equation (5)
            # ponemos las nuevas predicciones en la columna de w
            leaf.data[:,-1] = np.asanyarray(weight).reshape(-1,1) # forzar dimesniones 2D: (n,1)
            new_leafs.append(leaf.data)
        
        self.new_data = np.vstack(new_leafs)    

           


class XGBoost:
    def __init__(self, y, X,learning_rate = 0.1,boosting_rounds = 10, loss_function='mse'):
        super().__init__()
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.boosting_rounds = boosting_rounds
        self.y = y
        self.X = X
        self.row_id = np.arange(self.X.shape[0]).reshape(-1, 1)

        self.y_pred = None

    def get_gradients(self,y_pred):
        
        if self.loss_function == 'mse':
            # Gradiente y Hessiano para MSE
            g = (y_pred - self.y) # realmente aqui falta el 2 pero como es constante se ignora
            h = np.ones_like(self.y)
        
        elif self.loss_function == 'logistic':
            # Gradiente y Hessiano para Log Loss
            p = 1 / (1 + np.exp(-y_pred))  # Probabilidades predichas (Sigmoid)
            g = p - self.y  # Gradiente
            h = p * (1 - p)  # Hessiano
        else:
            raise NotImplementedError("Solo 'mse' y 'logistic' implementadas")
        
        return g, h
    
    def fit(self):

        y_pred = np.zeros_like(self.y)  # Inicializamos las predicciones
        g, h = self.get_gradients(y_pred)
        w = np.zeros_like(self.y) 

        data = np.concatenate((self.row_id, self.y, self.X, g, h, w), axis=1)
        for _ in range(self.boosting_rounds):

            # Comenzamos a hacer crecer los arboles
            model = Tree(data, eps=0.05, alpha=0.05, k=self.X.shape[1], max_depth=3)
            model.fit(data)
            model.get_leaves(model.root)
            model.do_predicttion(model.leaves)
           

            # Ahora reordenamos new_data para que coincida con el orden original
            # Aseguramos consistencia row_wise
            order = np.argsort(model.new_data[:, 0].astype(int))
            new_data = model.new_data[order]
            model.new_data = new_data

            # cogemos los nuevos pesos
            w = model.new_data[:,-1].reshape(-1, 1) # forzamos a 2D (n,1)
            # Añadimos nuevos pesos al Boosting y obtenemos nuevas predicciones
            y_pred += self.learning_rate*w 
            # recalculamos gradientes y hessianos
            g, h = self.get_gradients(y_pred) 

            #Construimos nueva base de datos para el siguiente arbol
            model.new_data[:,-3] = g.flatten() # mismo forzado a 2D
            model.new_data[:,-2] = h.flatten() # mismo forzado a 2D
            data = model.new_data
        
        self.y_pred = y_pred
        
        return y_pred                                


###   PRUEBA  ########################    

np.random.seed(42)

n = 1000  # Número de filas
k = 5    # Número de features (puedes cambiar esto a cualquier valor)

# Generamos datos aleatorios
X = np.random.rand(n, k)

# Generamos coeficientes aleatorios para cada feature
# Puedes modificar esto para tener coeficientes específicos
coeficientes = np.random.uniform(1, 5, size=(k, 1))
print(f"Coeficientes verdaderos: {coeficientes.flatten()}")

# Relación lineal: y = c1*X1 + c2*X2 + ... + ck*Xk + ruido
y = X @ coeficientes + 0.1 * np.random.randn(n, 1)

# Inicializamos y entrenamos el modelo
model = XGBoost(y, X, learning_rate=0.1, boosting_rounds=10, loss_function='mse')
predictions = model.fit()


print("Real vs Predicción:")
y_predict = np.hstack((y, predictions))
error = np.abs(y_predict[:,0] - y_predict[:,1])
y_predict = np.hstack((y_predict, error.reshape(-1,1)))
print(y_predict[:10])  # Muestra las primeras 10 predicciones vs valores reales
print("Error medio absoluto:", np.mean(error))