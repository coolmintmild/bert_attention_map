import pandas as pd
import numpy as np
import os

def to_csv(Item, relation_dict):
    print(f'{Item}')
    ItemPath = f'{Root}/attentions/{Item}'
    Files = os.listdir(f'{ItemPath}')

    for File in Files:
        FilePath = f'{ItemPath}/{File}'
        print(f'{FilePath}')
        df = pd.read_pickle(FilePath)

        for relation in relation_dict.keys():
            layer = relation_dict[relation][0]
            head = relation_dict[relation][1]
            df[relation] = np.array([atts[(layer-1)][(head-1)] for atts in df.Attentions])

        try:
            df.to_csv(f"c:/R/bertmap/{Item}.csv")
        except Exception as ex:
            print(ex)
            os.makedirs('c:/R/bertmap/')
            print("Created a new directory")
            df.to_csv(f"c:/R/bertmap/{Item}.csv")

Root = f'C:/python/bertmap'                                 # path for root folder
Items = os.listdir(f'{Root}/attentions')                    # path where pickles for attention weights are saved
relation_dict = {'dobj': (8, 10), 'nsubj': (8, 2)}          # {relation: (layer, head)

for Item in Items:
    to_csv(Item, relation_dict)