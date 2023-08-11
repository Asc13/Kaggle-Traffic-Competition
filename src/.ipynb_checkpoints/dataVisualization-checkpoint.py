import seaborn
import pandas
import numpy
import matplotlib.pyplot as plt

#Verificar outliers
def outliers(df):
    print(df.describe())
    
    plt.figure(figsize=(12,7))
    df.boxplot(vert=False)
    
    plt.show()

def corr_matrix(df):
    seaborn.heatmap(df.corr(), vmin = -1, vmax = 1, square = True, annot = True)
    plt.show()

def dataVisualization(df):
    #print(df[['record_date', 'month','day','week']].head())
    #outliers(df)
    #duplicates()
    corr_matrix(df)

    print("Variancia: ")
    print(df.var())
    print("Skewness: ")
    print(df.skew(axis=1))
    print(df)
    

