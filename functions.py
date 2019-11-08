import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score


def class_report(y_hat, y_hat2):
    """
        Esta función se encarga de graficar las métricas que 
        tomaré en consideración para saber la precisión de
        cada clase.
        
        Gráfica para:
            1.- f1_score
            2.- precision_score
            3.- recall_score
     """
    
    f1score = f1_score(y_hat, y_hat2, average=None)
    precision = precision_score(y_hat, y_hat2, average=None)
    recall = recall_score(y_hat, y_hat2, average=None)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(3, 1, 1)
    plt.barh([0, 1], [f1score[0], f1score[1]], color='blue')
    plt.title("f1-score")
    
    plt.figure(figsize=(10, 5))
    plt.subplot(3, 1, 2)
    plt.barh([0, 1], [precision[0], precision[1]], color='blue')
    plt.title("precision")
    
    plt.figure(figsize=(10, 5))
    plt.subplot(3, 1, 3)
    plt.barh([0, 1], [recall[0], recall[1]], color='blue')
    plt.title("recall")
    
    
    
