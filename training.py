import numpy as np
import pandas as pd
import os
import json
import math
import argparse
from random import shuffle
import torch
import torch.nn as nn
from models.ginconv import GINConvNet

from utils.utils import *
from utils.loss_functions import *
from utils.dataset import *

parser = argparse.ArgumentParser()
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


def run_training(dataset, run_name, params_file):
    with open(params_file) as fp:
        params = json.load(fp)
    for model in [GINConvNet]:
        if model.__name__ == params.get('model', 'GINConvNet'):
            modeling = model
    study_name = modeling.__name__ + '_'+ dataset.replace('/', '_') + '_' + run_name
    if not os.path.exists(study_name):
        os.makedirs(study_name)
    logging('\nrunning on '+ study_name, study_name)
    logging('Epochs: '+ str(params.get("epochs", 100)), study_name)
    logging('Learning rate: ' + str(params.get("lr",0.0005)), study_name)

    # Data
    train_loader = load_data(params, dataset)
    test_loader = load_data(params, dataset, test=True)

    # training the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_file_name = study_name+'/model_' + study_name + '.model'

    model = modeling(params) 
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.get("lr",0.0005))

    model.to(device)
    best_rmse=1000
    best_mse = 1000
    best_epoch = -1

    result_file_name = study_name+'/result_' + study_name  +  '.csv'
    model_par, hyperp_table = count_parameters(model)
    logging(hyperp_table, study_name)
    logging(f"Total Trainable Params: {model_par}", study_name)

    for epoch in range(params.get("epochs", 100)):
        logging('Training on {} samples...'.format(len(train_loader.dataset)), study_name)
        model.train()
        for batch_idx, data in enumerate(train_loader):
            data_mol = data[0].to(device)
            pro_1D = data[1].to(device)
            optimizer.zero_grad()
            output = model(data_mol, pro_1D)
            loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device))
            loss.backward()
            optimizer.step()
            if (batch_idx % params.get("log_interval",100)) == 0 or (batch_idx == len(train_loader)):
                logging('Train epoch: {} [{}/{} ({:.0f}%)]\tLossMSE: {:.6f}, LossRMSE: {:.6f}'.format(epoch,
                                                                                    (batch_idx+1) * len(pro_1D),
                                                                                    len(train_loader.dataset),
                                                                                    100. * (batch_idx+1) / len(train_loader),
                                                                                    loss.item(), math.sqrt(loss.item())), study_name)
        G,P, smiles, klifs = predict(model, device, test_loader, study_name)
        ret = [rmse(G,P), mse(G,P), pearson(G,P), spearman(G,P)]
        if ret[0] < best_rmse:
            torch.save({'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item()}, model_file_name)
            with open(result_file_name,'w') as f:
                f.write("%s\n" % 'rmse,mse,pearson,spearman')
                f.write("%s\n" % ','.join(map(str,ret)))
            best_epoch = epoch+1
            best_rmse = ret[0]
            best_mse = ret[1]
            logging('rmse improved at epoch ' + str(best_epoch) + '; best_rmse, best_mse: ' + str( best_rmse) + ', ' + str(best_mse), study_name)
        else:
            logging('rmse, mse, pearson: ' + str(ret[0])+ ', ' +  str(ret[1]) + ', ' +  str(ret[2]) + '; No improvement since epoch ' + str(best_epoch) + '; best_rmse, best_mse: ' + str(best_rmse)+ ', ' +  str(best_mse), study_name)
    with open(study_name+'/params.json', 'w') as fp:
        json.dump(params, fp)
        
if __name__ == "__main__":
    args = parser.parse_args()
    run_training(args.data, args.run, args.params)
    