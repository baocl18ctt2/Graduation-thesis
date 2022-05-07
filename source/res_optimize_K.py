from res_numba import RES, TEMPDIR
from metrics import (micro_f1, macro_f1, example_based_accuracy,
                     precision_at_1, precision_at_3, precision_at_5, evaluate, macro_f1_2)
from sklearn.metrics import hamming_loss
from sklearn.model_selection import train_test_split
from lcif import traverse_daat_candidates, instance_knn_predict, create_index_partition
# from scipy.sparse import load_npz
from utils import partial_sort_top_k, normalize_rows, load_data
from os import path
from scipy.optimize import (basinhopping, differential_evolution,
                            shgo, dual_annealing)
import numpy as np, argparse, json


def instance_knn_RES_obj(k_power, X_train, X_val, Y_train, Y_val, doc_at_a_time, max_w,
                         metric , k_rows, alpha):
    # shape = (Y_train.shape[0], X_val.shape[0])
        
    # similarity = res_simil_power_k(k_power, shape)
    similarity = RES(X_train, X_val, k_power)
    similarity = partial_sort_top_k(similarity, k_rows)

    if doc_at_a_time.nnz > 0:
        kNN = traverse_daat_candidates(X_val, similarity, doc_at_a_time, max_w,
                                    k_rows, 'RES', k_power)
    else:
        kNN = similarity
        
    Y_pred = instance_knn_predict(kNN, alpha, Y_train)

    metric_result = evaluate(Y_val, Y_pred, metric)
    
    if metric is not hamming_loss:
        metric_result = -metric_result
        
    return metric_result
    
def optimize_k_power(X_train, X_val, Y_train, Y_val,
                                    metric, optimize_algo, upper_bound,
                                    k_rows, alpha, dataset, metric_name, kwargs):
    X_train = normalize_rows(X_train)
    X_val = normalize_rows(X_val)
    
    phi = create_index_partition(X_train, 100)
    max_w = X_val.multiply(phi[2])
    max_w.sort_indices()
        
    x0 = np.zeros(4, np.float32)
    x0[0] = 1
    
    bounds = [(0., upper_bound)]*4
    if optimize_algo is basinhopping:
        kwargs['minimizer_kwargs']['bounds'] = bounds
    
    args = (X_train, X_val, Y_train, Y_val, phi[1], max_w, metric, k_rows, alpha)
    # args = (X_val, Y_train, Y_val, doc_at_a_time, max_w, metric)

    if optimize_algo is basinhopping:
        kwargs['minimizer_kwargs']['args'] = args
        kwargs['minimizer_kwargs']['bounds'] = bounds

        res = optimize_algo(instance_knn_RES_obj, x0=x0, **kwargs)
    elif optimize_algo is shgo:
        res = optimize_algo(instance_knn_RES_obj, bounds=bounds, args=args, **kwargs)
    else:
        res = optimize_algo(instance_knn_RES_obj, x0=x0, bounds=bounds,
                            args=args, **kwargs)

    np.savetxt(path.join(TEMPDIR, f'matrix/{dataset}-{metric_name}.txt'), res.x, '%f')
    print(f'best score: {res.fun}')
    
def create_argument_parser():
    parser = argparse.ArgumentParser()
    size_range = np.arange(0., 1., 0.1)
    
    parser.add_argument('filepath')
    parser.add_argument('name')
    parser.add_argument('metric')
    parser.add_argument('optimize_algo')
    parser.add_argument('kRows', type=int)
    parser.add_argument('rowPower', type=float)
    
    parser.add_argument('--ub', type=float, default=np.inf)
    parser.add_argument('--random_state', type=int, default=None)
    parser.add_argument('--train_size', type=np.float32, choices= size_range, default=0.75)
    parser.add_argument('--test_size', type=np.float32, choices= size_range, default=0.25)
    parser.add_argument('--shuffle', type=bool, choices=[True, False], default=False)
    parser.add_argument('--params')    
    
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
        raise ValueError('Unsupported metric')
    
    if args.optimize_algo == 'basinhopping':
        optimize_algo = basinhopping
    elif args.optimize_algo == 'differential_evolution':
        optimize_algo = differential_evolution
    elif args.optimize_algo == 'shgo':
        optimize_algo = shgo
    elif args.optimize_algo == 'dual_annealing':
        optimize_algo = dual_annealing
    else:
        raise ValueError('Unsupported optimization algorithm')
        
    if args.params is not None:
        with open(args.params) as fparams:
            kwargs = json.load(fparams)
    else:
        kwargs = {}
        
    if "mutation" in kwargs:
        kwargs["mutation"] = tuple(kwargs["mutation"])

    X, Y = load_data(args.filepath)
        
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y,
                                                      train_size=args.train_size, 
                                                     test_size=args.test_size,
                                                     random_state=args.random_state, shuffle=args.shuffle)

    
    print(f'Evaluation metric: {args.metric}')
    # optimize_k_power(metric, args.threshold, args.step)
    optimize_k_power(X_train, X_val, Y_train, Y_val, metric, optimize_algo, args.ub, args.kRows, args.rowPower,
                                    args.name, args.metric, kwargs)
    # optimize_k_power(metric, optimize_algo, args.ub, args.optimize_algo, kwargs)
    
