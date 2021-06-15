'''
Se supone que la última columna del dataframe es la que se quiere clasificar
Podría mejorarse en un futuro
'''

#%%
from math import log2
import pandas as pd

def getEntropyFromQuantities(listOfInt):
    total = sum(listOfInt)
    proportions = [value/total for value in listOfInt]
    proportionsMultipliedByLog2 = [p * log2(p) for p in proportions]
    sumatory = sum(proportionsMultipliedByLog2)
    return -sumatory

def getEntropy(dataset):
    classificationColumn = dataset.iloc[:, -1:]
    valueCounts = classificationColumn.value_counts()

    return getEntropyFromQuantities(list(valueCounts.values))

def informationGain(datasetDataFrame, attribute):
    values = datasetDataFrame[attribute].unique()
    
    summatory = 0
    for value in values:
        datasetFilteredByValue = datasetDataFrame[datasetDataFrame[attribute] == value]
        entropy = getEntropy(datasetFilteredByValue)
        proportionOverTotal = len(datasetFilteredByValue) / len(datasetDataFrame)
        summatory += proportionOverTotal * entropy
    
    return getEntropy(datasetDataFrame) - summatory

#%%
df = pd.read_csv('dataset.csv')

print('Entropy(S) = ', getEntropy(df))
print('Gain(S, Humidity) = ', informationGain(df, 'Humidity'))

# %%
assert(getEntropy(df) == 0.9402859586706311)
assert(informationGain(df, 'Humidity') == 0.15183550136234159)