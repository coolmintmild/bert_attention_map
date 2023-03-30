from bertmap.util import get_heatmaps
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def to_heatmap(Item, Cond):
    print(f'{Item}')
    ItemPath = f'{Root}/attentions/{Item}'
    Files = os.listdir(f'{ItemPath}')
    for File in Files:
        FilePath = f'{ItemPath}/{File}'
        print(f'{FilePath}')
        df = pd.read_pickle(FilePath)
        numHeads = len(df.Attentions.iloc[0][0])

        # heatmap of averaged attention weights across conditions
        df_mean = df.groupby(Cond)["Attentions"].agg([np.mean])

        for i, row in df_mean.iterrows():
            fig = get_heatmaps(row["mean"], numHeads)
            HeatmapFolder = f'/heatmaps/{Item}/{File[:-4]}'         # where the ouputfile is created.
            HeatmapFile = f'/{i}.png'
            try:
                fig.savefig(Root + HeatmapFolder + HeatmapFile)
            except Exception as ex:
                print(ex)
                os.makedirs(os.path.join(Root + HeatmapFolder))
                print("Created a new directory")
                fig.savefig(Root + HeatmapFolder + HeatmapFile)
            plt.close()


Root = f'C:/python/bertmap'                     # path for root folder
Items = os.listdir(f'{Root}/attentions')                    # path where pickles for attention weights are saved

# change conditions to be averaged by adjusting this:
# e.g. ["Dependency", "Length", "GP", "Cue"] | ["Dependency", "Length", "GP"]
Cond = ["Dependency", "Length", "GP", "Cue"]
for Item in Items:
    to_heatmap(Item, Cond)