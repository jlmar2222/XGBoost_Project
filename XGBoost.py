# Aqui vanmos a empezar a codificar el XGBoost


class XGBoost:
    def __init__(self, data):
        super().__init__()
        self.data = data    
    
        pass





def aprox_split_candidates(data): 
    # Esta función representa el Weighted Quantile Sketch
    # creado para encontrar candidatos de forma efectiva
    # sin tener que recorrer todos los posibles splits

    # Recibe un todos los feature (columna)
    # Return los G y H de lso candidatos 
    # --> son la suma de los g y h de las instancias dentro del quantile

    # recorre todas las columnas del data set --> all features
    for t in data:
        pass

    pass