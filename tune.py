""" Run hyperparameter tuning on validation set to determine optimal detection parameters. """

from utils.evaluate import *
import os
import h5py as h5
from models import SFANet, SFANetUnet
from utils.preprocess import *
import optuna
import yaml
import numpy as np
import json


import argparse
import sys, importlib
config_name = next((sys.argv[i + 1] for i, v in enumerate(sys.argv) if v == '--config' and i + 1 < len(sys.argv)), 'config')
config = importlib.import_module(config_name)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=config.prepared_data_path, help='path to data hdf5 file')
    parser.add_argument('--log', default=config.log_directory, help='path to log directory')
    parser.add_argument('--ntrials', type=int, default=100, help='number of trials')
    parser.add_argument('--max_distance', type=float, default=10, help='max distance from gt to pred tree (in pixels)')
    parser.add_argument('--config', type=str, default='config', help='Config module name (already loaded)')
    
    args = parser.parse_args()

    f = h5.File(args.data,'r')
    images = f['val/images'][:]
    gts = f['val/gt'][:]

    preds_path = os.path.join(args.log,'val_preds.npy')
    if os.path.exists(preds_path):
        print('----- loading predictions from file -----')
        preds = np.load(preds_path)
    else:
        bands = f.attrs['bands']
        preprocess = eval(f'preprocess_{bands}')

        model_type = config.model

        if model_type == 'vgg':
            model_builder = SFANet
        elif model_type == 'resnet':
            model_builder = SFANetRes
        elif model_type == 'efficientnet':
            model_builder = SFANetEfficient
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        training_model, model = model_builder.build_model(images.shape[1:], preprocess_fn=preprocess)

        weights_path = os.path.join(args.log, 'best.weights.h5')
        training_model.load_weights(weights_path)



        weights_path = os.path.join(args.log,'best.weights.h5')
        training_model.load_weights(weights_path)

        print('----- getting predictions from trained model -----')
        preds_list = []
        for i in range(len(images)):
            pred = model.predict(np.expand_dims(images[i], axis = 0), verbose = False)
            preds_list.append(pred[0, ..., 0])
        preds = np.array(preds_list)
        
        np.save(preds_path, preds)

        
        
    #### DETECTION TUNING ####
    def objective_detection(trial):
        min_distance = trial.suggest_int('min_distance', 1, 2)
        mode = 'abs'
        threshold_abs = trial.suggest_float('threshold_abs', 0, 1)
        threshold_rel = trial.suggest_float('threshold_rel', 0, 1)
        h_value = trial.suggest_float('h_value', 0, 0.5)

        results = evaluate(
            gts=gts,
            preds=preds,
            method=config.peak_method,
            min_distance=min_distance,
            threshold_rel= None,
            threshold_abs=threshold_abs if mode == 'abs' else None,
            max_distance=args.max_distance,
            h_value=config.h_max_value,
            return_locs=True
        )
        return 1 - results['fscore']

    print('----- running detection tuning -----')
    study_detection = optuna.create_study(direction='minimize')
    study_detection.optimize(objective_detection, n_trials=args.ntrials)
    best_params = study_detection.best_params
    
    print('Best detection params:', best_params)

    # Save detection params
    with open(os.path.join(args.log, 'params_detection.yaml'), 'w') as f:
        yaml.dump(best_params, f)
    
if __name__ == '__main__':
    main()
