import argparse, json, time, sys, re, matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from scipy.sparse import spmatrix, vstack
from sklearn import svm, tree, ensemble
from sklearn.tree import export_text, plot_tree


def get_coefficients(model: Union[svm.LinearSVC, tree.DecisionTreeClassifier], model_type: str, feature_names: list) -> pd.DataFrame:

    if model_type == 'svm':
        importances = model.coef_[0]
        importance = {f:i for f, i in zip(feature_names, importances)}
        sorted_list = sorted(importance.items(), key = lambda item:item[1], reverse=True)
        n_first, n_last = sorted_list[:25], sorted_list[-25:]
        sorted_importance = {f:i for f, i in n_first+n_last}
    if model_type == 'dt':
        importances = model.feature_importances_
        importance = {f:i for f, i in zip(feature_names, importances)}
        sorted_importance = {f: i for f, i in sorted(importance.items(), key = lambda item:item[1])[-50:]}
        
    cols = ['Importances']

    coefs = pd.DataFrame.from_dict(sorted_importance, orient='index', columns=cols)

    return coefs


def plot_feature_importance(df: pd.DataFrame, model_path: str) -> None:

    # filter zeros for better readability
    zeros = False
    i = 0
    for imp_value in df.iloc:
        if df.iloc[i]['Importances'] == 0.0:
            if not zeros:
                print('not zeros')
                df = df.rename(index={imp_value.name: '…'})
                zeros = True
            else:
                print('zeros')
                df = df.drop([imp_value.name])
                i -= 1
        i += 1

    df.plot.barh(figsize=(7, 10))
    plt.title('{} feature weights'.format(' '.join(model_path.split('/')[-1].split('_')[:-1])))
    
    plt.axvline(x=0, color=".5")
    plt.xlabel('Raw weights values')
    plt.subplots_adjust(left=0.3)
    
    plt.savefig('plots/feature_importances/{}_feature_importances.png'.format(model_path.split('/')[-1]))
