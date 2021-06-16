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

def getInformationGain(datasetDataFrame, attribute):
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

def desicionTree(dataset, attributes):
    '''
    Trees have the form of:
    (node, [(edge, tree), ...])
    '''
    if not attributes:
        return 'Out of attribures'

    maximumInformationGain = -9999
    for attribute in attributes:
        informationGain = getInformationGain(dataset, attribute)

        if informationGain > maximumInformationGain:
            maximumInformationGain = informationGain
            bestAttribute = attribute
    
    # Values of the attribute, edges of the tree
    values = list(dataset[bestAttribute].unique())
    nextDatasets = [dataset[dataset[bestAttribute] == value] for value in values]

    nextAttributes = [attribute for attribute in attributes if attribute != bestAttribute]

    nextDesicionTrees = []
    for value, nextDataset in zip(values, nextDatasets):
        classifications = list(nextDataset[nextDataset.columns[-1]].unique())
        if len(classifications) == 1:
            nextDesicionTrees.append(classifications)
        else:
            nextDesicionTree = desicionTree(nextDataset, nextAttributes)
            nextDesicionTrees.append(nextDesicionTree)

    return (bestAttribute, nextDesicionTrees)


#%%
df = pd.read_csv('dataset.csv')
print(df)

#%%
dt = desicionTree(df, list(df.columns)[:-1])
from pprint import pprint
pprint(dt)
#%%
assert(getEntropy(df) == 0.9402859586706311)
assert(getInformationGain(df, 'Humidity') == 0.15183550136234159)
