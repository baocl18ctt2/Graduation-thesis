import numpy as np
from sklearn.metrics import hamming_loss, f1_score, precision_score
from threshold import apply_threshold, find_optimal_threshold
from utils import get_indicator, cardinality, partial_sort_top_k
from numba import njit
# from numba.typed import List

@njit
def _accuracy(Y_indices, Y_indptr, Yp_indices, Yp_indptr):
    '''
    Hàm tính tổng tỷ lệ phần giao và phần hợp
    giữa tập nhãn dự đoán và tập nhãn đúng trên các đối tượng.
    
    Đầu vào:
    --------
    Y_indices, Y_indptr, Yp_indices, Yp_indptr: np.ndarray
        mảng một chiều thể hiện nhãn và phân đoạn cho từng đối tượng (dòng)
        của tập nhãn đúng và nhãn dự đoán.
        
    Đầu ra:
    -------
    acc: float
        tổng tỷ lệ phần giao và hợp giữa nhãn đúng và dự đoán.
    '''
    acc = 0
    for y_start, y_end, yp_start, yp_end in zip(Y_indptr, Y_indptr[1:],Yp_indptr, Yp_indptr[1:]):
        y = Y_indices[y_start:y_end]
        y_pred = Yp_indices[yp_start:yp_end]

        if y.size ==  y_pred.size == 0:
            acc += 1
        else:
            intersect = np.intersect1d(y, y_pred).size
            union = y.size + y_pred.size - intersect

            acc += intersect / union

    return acc

def example_based_accuracy(Y, Y_pred):
    '''
    Hàm tính độ chính xác của dự đoán
    theo trung bình tỷ lệ phần giao trên phần hợp của các đối tượng.
    
    Đầu vào:
    --------
    Y, Y_pred: scipy.sparse.csr_matrix
        ma trận gồm các véc-tơ nhãn đúng và dự đoán.
        
    Đầu ra:
    -------
    acc: float
        trung bình tỷ lệ phần giao trên phần hợp của các đối tượng.
    '''
    # acc = 0
    # for y_start, y_end, Y_pred_start, Y_pred_end in zip(y.indptr, y.indptr[1:],
    #                                                    Y_pred.indptr, Y_pred.indptr[1:]):
    #     # y_start, y_end = y.indptr[i:i+2]
    #     # Y_pred_start, Y_pred_end = Y_pred.indptr[i:i+2]
        
    #     intersect = np.intersect1d(y.indices[y_start:y_end],
    #                               Y_pred.indices[Y_pred_start:Y_pred_end],
    #                               assume_unique=True)
        
    #     union = np.union1d(y.indices[y_start:y_end],
    #                        Y_pred.indices[Y_pred_start:Y_pred_end])
    #     if union.size == 0:
    #         acc += 1
    #     else:
    #         acc += intersect.size / union.size
    acc = _accuracy(Y.indices, Y.indptr, Y_pred.indices, Y_pred.indptr)
    return acc / Y.shape[0]

# def hamming_loss(y, Y_pred):
#     # i, loss = 0, 0
#     loss = 0
#     # while i < y.shape[0]:
#     for y_start, y_end, Y_pred_start, Y_pred_end in zip(y.indptr, y.indptr[1:],
#                                                        Y_pred.indptr, Y_pred.indptr[1:]):
#         # y_start, y_end = y.indptr[i:i+2]
#         # Y_pred_start, Y_pred_end = Y_pred.indptr[i:i+2]
#         intersect = np.intersect1d(y.indices[y_start:y_end],
#                                   Y_pred.indices[Y_pred_start:Y_pred_end],
#                                   assume_unique=True)
        
#         loss_instance = (y.indices[y_start:y_end].size + Y_pred.indices[Y_pred_start:Y_pred_end].size - 2 * intersect.size) / y.shape[1]
#         loss += loss_instance
#         # i += 1
    
#     return loss / y.shape[0]

@njit
def _confusion_matrix(Y_indices, Y_indptr, Yp_indices, Yp_indptr, tp, fp, fn):
    '''
    Hàm tính các giá trị dự đoán đúng, phủ định sai và nhận nhãn sai cho các nhãn
    
    Đầu vào:
    --------
    Y_indices, Y_indptr, Yp_indices, Yp_indptr: np.ndarray
        mảng một chiều thể hiện nhãn và phân đoạn cho từng đối tượng (dòng)
        của tập nhãn đúng và nhãn dự đoán.
        
    tp, fp, fn: np.ndarray
        mảng một chiều lưu các giá trị True Positive, False Positive và False Negative
    '''
    for y_start, y_end, yp_start, yp_end in zip(Y_indptr, Y_indptr[1:],Yp_indptr, Yp_indptr[1:]):
        y = set(Y_indices[y_start:y_end])
        y_pred = set(Yp_indices[yp_start:yp_end])

        for item in y.intersection(y_pred):
            tp[item] += 1

        for item in y.difference(y_pred):
            fn[item] += 1

        for item in y_pred.difference(y):
            fp[item] += 1

def compute_confusion_matrix(Y, Y_pred):
    tp = np.zeros(Y.shape[1], np.float32)
    fp = np.zeros(Y.shape[1], np.float32)
    fn = np.zeros(Y.shape[1], np.float32)
    
    # for y_start, y_end, Y_pred_start, Y_pred_end in zip(y.indptr, y.indptr[1:],
    #                                                    Y_pred.indptr, Y_pred.indptr[1:]):
        
    #     intersect = np.intersect1d(y.indices[y_start:y_end],
    #                               Y_pred.indices[Y_pred_start:Y_pred_end],
    #                               assume_unique=True)
    #     tp[intersect] += 1
        
    #     y_xor_yhat = np.setdiff1d(y.indices[y_start:y_end],
    #                               Y_pred.indices[Y_pred_start:Y_pred_end],
    #                               assume_unique=True)
        
    #     fn[y_xor_yhat] += 1
        
    #     yhat_xor_y = np.setdiff1d(Y_pred.indices[Y_pred_start:Y_pred_end],
    #                               y.indices[y_start:y_end],
    #                               assume_unique=True)
        
    #     fp[yhat_xor_y] += 1
    _confusion_matrix(Y.indices, Y.indptr, Y_pred.indices, Y_pred.indptr, tp, fp, fn)
        
    return tp, fp, fn

def precision(tp, fp, method=''):
    if method == '':
        #trả về mảng precision cho từng nhãn
        tp_plus_fp = tp + fp
        return np.divide(tp, tp_plus_fp, out=np.zeros_like(tp), where=tp_plus_fp!=0)
    
    elif method == 'micro':
        tp_sum = tp.sum(); fp_sum = fp.sum()
        if (tp_sum + fp_sum) == 0:
            return 0
        else:
            return tp_sum / (tp_sum + fp_sum)
        
    elif method == 'macro':
        return np.nanmean(tp / (tp + fp))
    
    else:
        raise ValueError('method: invalid method')

def recall(tp, fn, method=''):
    if method == '':
        #trả về mảng recall cho từng nhãn
        tp_plus_fn = tp + fn
        return np.divide(tp, tp_plus_fn, out=np.zeros_like(tp), where=tp_plus_fn!=0) 
    
    elif method == 'micro':
        tp_sum = tp.sum(); fn_sum = fn.sum()
        if (tp_sum + fn_sum) == 0:
            return 0
        else:
            return tp_sum / (tp_sum + fn_sum)
        
    elif method == 'macro':
        return np.nanmean(tp / (tp + fn))
    
    else:
        raise ValueError('method: invalid method')
    
@njit
def _macrof1(tp, fp, fn):
    not_occurring, totalF1 = 0, 0
    for tp_label, fp_label, fn_label in zip(tp, fp, fn):
        if (tp_label + fn_label == 0) or (tp_label + fp_label + fn_label == 0):
            not_occurring += 1
            continue

        if tp_label + fp_label == 0:
            prec = 0
        else:
            prec = tp_label / (tp_label + fp_label)

        rec = tp_label / (tp_label + fn_label)

        if prec + rec == 0:
            f1 = 0
        else:
            f1 = 2 * prec * rec / (prec + rec)

        totalF1 += f1
    
    return totalF1 / (tp.size - not_occurring)
        # prec = precision(tp, fp)
        
        # rc = recall(tp, fn)
        # prec_plus_rc = prec + rc
        
        # f1 = np.divide(2*prec*rc, prec_plus_rc,out=np.zeros_like(prec_plus_rc), 
        #                where=prec_plus_rc!=0, dtype=np.float64)
        
        # return f1.mean()
    
# def micro_f1(tp, fp, fn):
#     return f1(tp, fp, fn, 'micro')

def macro_f1_2(Y_true, Y_pred):
    tp, fp, fn = compute_confusion_matrix(Y_true, Y_pred)
    return _macrof1(tp, fp, fn)

def compare_metric_score(current_best, candidate, metric):
    return ((metric is hamming_loss) and (candidate < current_best)) or \
            (candidate > current_best)

def micro_f1(Y_true, Y_pred):
    return f1_score(Y_true, Y_pred, average='micro', zero_division=1)

def macro_f1(Y_true, Y_pred):
    return f1_score(Y_true, Y_pred, average='macro', zero_division=1)

def precision_at_k(Y_true, Y_pred, k):
    y_k_pred = Y_pred.copy()
    partial_sort_top_k(y_k_pred, k)
    
    Y_pred_indicator = get_indicator(y_k_pred)
    return precision_score(Y_true, Y_pred_indicator, average='samples')

def precision_at_1(Y_true, Y_pred):
    return precision_at_k(Y_true, Y_pred, 1)

def precision_at_3(Y_true, Y_pred):
    return precision_at_k(Y_true, Y_pred, 3)

def precision_at_5(Y_true, Y_pred):
    return precision_at_k(Y_true, Y_pred, 5)

# def precision_at_all_k(Y_true, Y_pred):
#     return [precision_at_k(Y_true, Y_pred, k) for k in [1,3,5]]

def evaluate(Y_true, Y_pred, metric):
    '''
    Đánh giá tập dự đoán với giá trị đúng theo độ đo được chọn.
    
    Đầu vào:
    --------
    Y_true, Y_pred: scipy.sparse.csr_matrix
        tập nhãn đúng và dự đoán của đối tượng.
        
    metric: callable
        độ đo đánh giá
        
    Đầu ra:
    -------
    metric_result: float
        điểm đánh giá tập dự đoán so với tập nhãn đúng
        theo độ đo được chọn.
    '''
    if Y_true.shape != Y_pred.shape:
            raise ValueError('dimension mismatch')
            
    Y_true = Y_true.tocsr()
    Y_pred = Y_pred.tocsr()
    
    if metric not in {precision_at_1, precision_at_3, precision_at_5}:
        lcard = cardinality(Y_true)
        t = find_optimal_threshold(lcard, Y_pred)
        at_least_one_score = np.min(np.diff(Y_true.indptr)) > 0
        apply_threshold(Y_pred, t, at_least_one_score)
        
        Y_pred_indicator = get_indicator(Y_pred)  
        result = metric(Y_true, Y_pred_indicator)        
    else:
        result = metric(Y_true, Y_pred)
    return result
