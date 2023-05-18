"""Testing affinity models."""
from ssl import PROTOCOL_TLSv1_2
import numpy as np
import pandas as pd
import sys, os
import math
import argparse
import json
from random import shuffle
import torch
import torch.nn as nn
from models.ginconv import GINConvNet

from utils.utils import *
from utils.loss_functions import *
from utils.dataset import *
parser = argparse.ArgumentParser()

parser.add_argument(
    "model_dir",
    type=str,
    help="model directory"
)
   
parser.add_argument(
    "run",
    type=str,
    help="Name of the run."
)
parser.add_argument(
    "data",
    type=str,
    help="Name of the data."
)
parser.add_argument(
    "params",
    type=str,
    help="Config file."
)


def run_testing(model_dir, dataset, run_name, params_file):
    with open(params_file) as fp:
        params = json.load(fp)
    for model in [GINConvNet]:
        if model.__name__ == params.get('model', 'GINConvNet'):
            modeling = model
    study_name  = modeling.__name__+ '_'+ dataset.replace('/', '_') + '_' + run_name
    logging('\ntesting on '+ study_name, study_name)
    test_loader = load_data(params, dataset, test=True)
    device = torch.device('cpu')
    model = modeling(params)
    checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    G,P, smiles, klifs = predict(model, device, test_loader, study_name)
    test_results = {'pIC50': list(G), 'prediction': list(P), 'smiles': list(smiles), 'klifs': list(klifs) }
    test_results = pd.DataFrame(data = test_results)
    test_results.to_csv(os.path.dirname(model_dir) +'/'+ dataset.replace('/', '_') +'_test_predictions.csv') 
    
    ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P)]

    result_file_name = os.path.dirname(model_dir)+ '/' + dataset.replace('/', '_') +  '_test_results.csv'
    with open(result_file_name,'w') as f:
        f.write("%s\n" % 'rmse,mse,pearson,spearman')
        f.write("%s\n" % ','.join(map(str,ret)))
        f.close()
        



if __name__ == "__main__":
    args = parser.parse_args()
    run_testing(args.model_dir, args.data, args.run, args.params)
    