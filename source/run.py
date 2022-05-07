from lcif import run_lcif, run_ins_fast, run_ins_search, run_feat_knn
from utils import load_data, compare_n_resize
from metrics import (micro_f1, macro_f1, precision_at_5,
                    example_based_accuracy, hamming_loss)
import sys, argparse, time

def create_argument_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('train_path')
    parser.add_argument('test_path')
    parser.add_argument('measure', choices=['cosine', 'RES'])
    parser.add_argument('metric')
    parser.add_argument('model', choices=['ins_search', 'ins_fast', 'feat', 'lcif'])
    
    parser.add_argument('-k', type=int)
    parser.add_argument('-a', type=float)
    parser.add_argument('-b', type=float)
    parser.add_argument('-l', type=float)
    
    parser.add_argument('--returnPred', type=bool, choices=[True, False], default=False)
    parser.add_argument('--evaluateAll', type=bool, choices=[True, False], default=True)
    parser.add_argument('--kPower', nargs=4)
    
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
    elif args.metric == 'precision@5':
        metric = precision_at_5
    else:
        raise ValueError('Invalid metric')
    
    X_train, Y_train = load_data(args.train_path)
    X_test, Y_test = load_data(args.test_path)
    
    compare_n_resize(X_train, X_test)
    compare_n_resize(Y_train, Y_test)

    start_time = time.time()
    
    if args.model == 'ins_search':
        results = run_ins_search(X_train, X_test, Y_train, Y_test,
                                args.k, args.a, metric, args.returnPred,
                                args.measure, args.kPower)
    elif args.model == 'ins_fast':
        results = run_ins_fast(X_train, X_test, Y_train, Y_test,
                              args.k, args.a, metric, args.returnPred,
                            args.measure, args.kPower)
    else:
        results = run_lcif(X_train, X_test, Y_train, Y_test,
                        args.kRow, args.alpha, args.beta, args.lc, metric,
                        args.returnPred, args.evaluateAll,
                        args.measure, args.kPower)

    print("--- %s seconds ---" % (time.time() - start_time))
    print(results)
    