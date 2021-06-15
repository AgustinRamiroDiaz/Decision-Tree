'''
Se supone que la última columna del dataframe es la que se quiere clasificar
Podría mejorarse en un futuro
'''

#%%
from math import log2
import pandas as pd

def getEntropyFromQuantities(listOfInt):
    log = f'E{tuple(listOfInt)} = '
    total = sum(listOfInt)
    proportions = [value/total for value in listOfInt]
    proportionsMultipliedByLog2 = [p * log2(p) for p in proportions]

    sumatory = sum(proportionsMultipliedByLog2)

    log += ' + '.join([f'{round(proportion, 2)} * log_2({round(proportion, 2)})' for proportion in proportions]) + f'\n= {round(sumatory, 2)}'
    print(log)
    return -sumatory

def getEntropy(dataset):
    classificationColumn = dataset.iloc[:, -1:]
    valueCounts = classificationColumn.value_counts()

    return getEntropyFromQuantities(list(valueCounts.values))

def informationGain(datasetDataFrame, attribute):
    totalEntropy = getEntropy(datasetDataFrame)
    log = f'Gain(S, {attribute}) = E(S) - sum( |Sv| / |S| * E(Sv) )\n= {round(totalEntropy, 2)}'
    values = datasetDataFrame[attribute].unique()
    
    summatory = 0
    for value in values:
        datasetFilteredByValue = datasetDataFrame[datasetDataFrame[attribute] == value]
        entropy = getEntropy(datasetFilteredByValue)
        proportionOverTotal = len(datasetFilteredByValue) / len(datasetDataFrame)
        summatory += proportionOverTotal * entropy
        log += f' - {round(proportionOverTotal, 2)} * {round(entropy, 2)}'
    
    result = totalEntropy - summatory
    log += f'\n= {round(result, 2)}'
    print(log)
    return result

#%%
df = pd.read_csv('dataset.csv')

assert(getEntropy(df) == 0.9402859586706311)
assert(informationGain(df, 'Humidity') == 0.15183550136234159)