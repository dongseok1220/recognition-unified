from tqdm import tqdm
import pandas as pd


def unify(handclap, headbanging, sitstand):
    i = j = k = 0
    unified = {}
    merged_list = sorted(set(handclap + headbanging + sitstand))
    for idx in tqdm(merged_list):
        val_i = val_j = val_k = False
        if i < len(handclap) and handclap[i] == idx:
            i += 1
            val_i = True
        if j < len(headbanging) and headbanging[j] == idx:
            j += 1
            val_j = True
        if k < len(sitstand) and sitstand[k] == idx:
            k += 1
            val_k = True
        val = (val_i, val_j, val_k)
        if (val_i or val_j or val_k):
            unified[idx] = val
    
    print(unified)
    return dict_to_dataframe(unified)



def dict_to_dataframe(data_dict):
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    df.columns = ['handclap', 'headbanging', 'sitstand']
    df.index.name = 'idx'
    return df.sort_index()