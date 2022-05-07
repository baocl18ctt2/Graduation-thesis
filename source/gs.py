from lcif import run_lcif_gs_extreme, run_lcif_gs_large
from utils import load_data
from sklearn.metrics import hamming_loss
from metrics import (micro_f1, macro_f1, example_based_accuracy,
                     precision_at_1, precision_at_3, precision_at_5, macro_f1_2)
from numpy import loadtxt
import argparse, warnings, time, utils

def create_argument_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('train_path')
    parser.add_argument('test_path')
    parser.add_argument('mode', choices=['GS', 'GS_BIG', 'ins'])
    parser.add_argument('measure', choices=['cosine', 'RES'])
    parser.add_argument('metric')
    
    parser.add_argument('--returnParams', type=bool, choices=[True, False], default=True)
    parser.add_argument('--evaluateAll', type=bool, choices=[True, False], default=True)
    parser.add_argument('--kPower')
    parser.add_argument('--knnFast', type=bool, choices=[True, False], default=True)
    
    return parser

if __name__ == '__main__':
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if args.metric == 'exampleBasedAccuracy':
        metric = example_based_accuracy
    elif args.metric == 'hammingLoss':
        metric = hamming_loss
    elif args.metric == 'microF1':
        metric = micro_f1
    elif args.metric == 'macroF1':
        metric = macro_f1
    elif args.metric == 'macroF1_2':
        metric = macro_f1_2
    elif args.metric == 'precision@1':
        metric = precision_at_1
    elif args.metric == 'precision@3':
        metric = precision_at_3
    elif args.metric == 'precision@5':
        metric = precision_at_5
    else:
        raise ValueError('Invalid metric')
        
    X_train, Y_train = load_data(args.train_path)
    X_test, Y_test = load_data(args.test_path)
    
    if args.mode == 'GS_BIG':
        utils.GS_BIG = True
    
    if isinstance(args.kPower, str):
        k_power = loadtxt(args.kPower)
    else:
        k_power = None
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        start_time = time.time()
        if args.mode == 'GS_BIG':
            params, result = run_lcif_gs_extreme(X_train, X_test, Y_train, Y_test, metric,
                                                args.returnParams, args.evaluateAll,
                                                args.measure, k_power, args.knnFast)
        else:
            params, result = run_lcif_gs_large(X_train, X_test, Y_train, Y_test, metric,
                                               args.returnParams, args.evaluateAll,
                                               args.measure, k_power, args.knnFast)
            
        print("--- %s seconds ---" % (time.time() - start_time))
    if args.returnParams:
        print(params)
        
    print('{:<15}\t{:<20}\t{:<20}\t{:<20}'.format('metric', 'INS. KNN', 'FEAT. KNN', 'LCIF'))
    
    print(f'{args.metric:<15}\t{result[0]:<20.7}\t{result[1]:<20.7}\t{result[2]:<20.7}')
