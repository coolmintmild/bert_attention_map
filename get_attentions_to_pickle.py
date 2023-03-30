from bertmap.util import load_model, get_attentions
import pandas as pd
import os

def get_attention(architecture='bert', version='bert-base-uncased'):
    print(f'{version} loading')
    model, tokenizer, numLayers, numHeads = load_model(architecture, version, cuda=True)
    for Item in Items:
        ItemPath = f'{Root}/items/{Item}'
        df = pd.read_excel(ItemPath, index_col=0)
        targets = []
        for i, row in df.iterrows():
            target_attentions = get_attentions(model, tokenizer, row, numLayers, numHeads, cuda=True)
            targets.append(target_attentions)
        df["Attentions"] = targets
        try:
            df.to_pickle(f'{Root}/attentions/{Item[:-5]}/{version}.pkl')            # where the ouput pickle file is created.
            print(f'{Item[:-5]} exported')
        except Exception as ex:
            print(ex)
            os.makedirs(os.path.join(f'{Root}/attentions/{Item[:-5]}'))
            print("Created a new directory")
            df.to_pickle(f'{Root}/attentions/{Item[:-5]}/{version}.pkl')
            print(f'{Item[:-5]} exported')

Root = f'C:/python/bertmap'                             # path for root folder
Items = os.listdir(f'{Root}/items')                    # path where items are saved
get_attention()

'''
architecture=['bert', 'albert']
version=['bert-base-uncased','bert-large-uncased','albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2','albert-xxlarge-v2']
'''




