import numpy as np
from utils import cardinality

def apply_threshold(y_pred, t, at_least_one_score=False):
    '''
    Hàm loại bỏ những nhãn trong tập dự đoán y_pred
    có điểm số thấp hơn hoặc bằng ngưỡng t.
    
    Đầu vào:
    --------
    y_pred: scipy.sparse.csr_matrix
        tập dự đoán.
        
    t: float
        ngưỡng điểm dự đoán.
        
    at_least_one_score: bool, mặc định=False
        đảm bảo có ít nhất 1 nhãn dự đoán
        trong trường hợp tất cả nhãn có điểm số <= t.
        
    Các thay đội được thực hiện ngay trên y_pred.
    '''
    if not at_least_one_score:
        y_pred.data[y_pred.data <= t] = 0
    else:
        # i = 0
        # while i < y_pred.shape[0]:
        for start, end in zip(y_pred.indptr, y_pred.indptr[1:]):
            # start, end = y_pred.indptr[i:i+2]
            data = y_pred.data[start:end]
            
            above_t_id = data > t
            if np.sum(above_t_id) == 0:
                # y_pred.data[start+1:end] = 0
                data[1:] = 0
            else:
                # y_pred.data[~above_t_id] = 0
                # rm_indices = (~above_t_id).nonzero()[0] + start
                # y_pred.data[rm_indices] = 0
                data[~above_t_id] = 0
        
    y_pred.eliminate_zeros()


def _find_threshold(labels_card, y, thresholds):
    y = y.copy()
    best_t, best_diff = 0, np.inf
    
    for t in thresholds:
        apply_threshold(y, t)
        y_card = cardinality(y)
        card_diff = abs(y_card - labels_card)
        if card_diff < best_diff:
            best_diff = card_diff
            best_t = t
            
    return best_t

def find_optimal_threshold(label_cardinality, y_pred):
    '''
    Hàm tìm ngưỡng điểm dự đoán tốt nhất cho tập dự đoán.
    
    Đầu vào:
    --------
    label_cardinality: float
        số lượng nhãn trung bình ở mỗi đối tượng của tập nhãn đúng.
        
    y_pred: scipy.sparse.csr_matrix
        tập nhãn dự đoán.
        
    Đầu ra:
    -------
    best_t: float
        giá trị ngưỡng tốt nhất.
    '''
#     label_cardinality = cardinality(labels)
    thresholds = np.arange(0, 1, .1)
    
    best_t = _find_threshold(label_cardinality, y_pred, thresholds)
    
    t_finegrained = np.arange(best_t-0.05, best_t+0.06, 0.01)
    best_t = _find_threshold(label_cardinality, y_pred, t_finegrained)
    
    return best_t