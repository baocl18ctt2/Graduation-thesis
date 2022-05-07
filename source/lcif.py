import numpy as np, warnings, utils
from scipy.sparse import csr_matrix, dok_matrix
from utils import (partial_sort_top_k, normalize_features, cardinality,
                   normalize_rows, compare_n_resize)
from metrics import hamming_loss, evaluate, compare_metric_score
from mpire import WorkerPool
from multiprocessing.sharedctypes import Array, Value
from ctypes import c_bool
from res_numba import RES
from sklearn.model_selection import KFold
from os import cpu_count
from numba import njit, prange

def compute_similarity(Xq, X=None, method='cosine', K=None, Xt=None):
    if method == 'cosine':
        if Xt is not None:
            simil = Xq.dot(Xt)
        else:
            simil = Xq.dot(X.transpose())
            
    elif method == 'RES':
        # simil = resonance_similarity(X, Xq, K)
        simil = RES(X, Xq, K)
        
    return simil

def instance_knn_search(Xq, k, X, simil_method='cosine', k_power=None):
    '''
    Hàm tìm k lân cận gần nhất cho đối tượng Xq trong tập dữ liệu.
    
    Đầu vào:
    --------
    Xq: scipy.sparse.
        tập đặc trưng cửa đối tượng truy vấn.
        
    k: int
        số lân cận gần nhất.
        
    X: scipy.sparse.csr_matrix
        tập đặc trưng của dữ liệu huấn luyện.
        
    similarity: scipy.sparse.csr_matrix
        ma trận độ tương đồng của các đối tượng truy vấn
        và các đối tượng huấn luyện.
        Thường dùng khi tìm kiếm tham số k tốt nhất, không phải tính lại
        độ tương đồng cho cùng tập huấn luyện và truy vấn nhiều lần.
        
    Đầu ra:
    -------
    kNN: scipy.sparse.csr_matrix
        ma trận ghi k đối tượng huấn luyện gần nhất và độ tương đồng với từng đối tượng truy vấn.
    '''
    # similarity = Xq.dot(X.transpose())
    if utils.GS_BIG:
        samples = 100
        if Xq.shape[0] > 10000:
            samples = 10000
        elif Xq.shape[0] > 1000:
            samples = 1000
            
        similarity = dok_matrix((Xq.shape[0], X.shape[0]))
        
        row_indptr = [0]
        for i in range(Xq.shape[0] // samples):
            row_indptr.append(row_indptr[i] + samples)
            
        mod = Xq.shape[0] % samples
        if mod != 0:
            row_indptr.append(row_indptr[-1] + mod)
            
        Xt = X.transpose()
        for start, end in zip(row_indptr, row_indptr[1:]):
            part_simil = compute_similarity(Xq[start:end], method=simil_method, K=k_power, Xt=Xt)
            similarity[start:end] = partial_sort_top_k(part_simil, k)
            
        kNN = similarity.tocsr()
    else:
        similarity = compute_similarity(Xq, X, simil_method, k_power)
        kNN = partial_sort_top_k(similarity, k)
        
    return kNN
    
    # partial_sort_top_k(similarity, k)
    # return similarity

def instance_knn_predict(kNN, alpha, labels):
    '''
    Dự đoán điểm tin cậy các nhãn cho đối tuợng truy vấn Xq dựa vào các lân cận gần nhất.
    
    Đầu vào: 
    --------
    kNN: scipy.sparse.csr_matrix
        ma trận ghi k đối tượng huấn luyện gần nhất và độ tương đồng với từng đối tượng truy vấn.
        
    alpha: int
        hệ sô hàm biến đổi mũ.
        
    labels: scipy.sparse.csr_matrix
        tập nhẫn của các đối tượng huấn luyện.
        
    Đầu ra: 
    -------
    Y_pred: scipy.sparse.csr_matrix
        mảng điểm dự đoán các nhãn cho Xq.
    '''
    kNN = kNN.power(alpha)
    kNN_sum = kNN.sum(1)
    inv_kNN_sum = np.divide(1., kNN_sum, where=kNN_sum!=0)
    
    Y_pred = kNN.dot(labels).multiply(inv_kNN_sum)
    return csr_matrix(Y_pred)


def create_simil_matrix(features, labels):
    '''
    Tính độ tương đồng giữa tất cả các đặc trưng và nhãn trong tập dữ liệu.
    Độ tương đồng được sử dụng cho thuật toán k lân cận gần nhất dựa trên các đặc trưng.
    
    Đầu vào:
    --------
    features: scipy.sparse.csr_matrix
        ma trận lưu trữ các đặc trưng của đối tượng trong tập dữ liệu.
        
    labels: scipy.sparse.csr_matrix
        ma trận lưu trữ các nhãn của đối tượng trong tập dữ liệu.
        
    Đầu ra: 
    -------
    S: scipy.sparse.csr_matrix
        ma trận độ tương đồng giữa các đặc trưng và các nhãn.
    '''
    S = features.transpose().dot(labels)
    
    label_sqrt_sum = np.sqrt(labels.sum(0))
    label_inv_norm = np.divide(1., label_sqrt_sum, where=label_sqrt_sum!=0)
    
    return S.multiply(label_inv_norm).tocsr()
    # simil = S.multiply(label_inv_norm).tocsr()
    # partial_sort_top_k(simil, 10)
    # return simil
    

def feature_knn_predict(Xq, S, beta, store_k):
# def feature_knn_predict(Xq, S, beta):
    '''
    Tính điểm tin cậy để dự đoán nhãn cho đối tượng truy vấn dựa vào các đặc trưng trong tập dữ liệu.
    
    Đầu vào:
    --------
    Xq: scipy.sparse
        tập đặc trưng của đối tượng truy vấn.
    S: scipy.sparse.csr_matrix
        ma trận độ tương đồng giữa các đặc trưng và nhãn trong tập dữ liệu.
    beta: float
        hệ số mũ
        
    Đầu ra: 
    -------
    Y_pred: scipy.sparse.csr_matrix
        điểm dự đoán các nhãn của đối tượng truy vấn.
    '''
    scores = Xq.dot(S.power(beta))
    
    Xq_sum = Xq.sum(1)
    inv_Xq_sum = np.divide(1., Xq_sum, where=Xq_sum!=0)
    scores = scores.multiply(inv_Xq_sum)
    
    Y_pred = partial_sort_top_k(scores, store_k)
    return Y_pred
    # partial_sort_top_k(scores, store_k)
    # return scores


def lcif(X_train, Y_train, X_test, k_rows, alpha, beta, lamb, return_all,
        simil_method='cosine', k_power=None, knn_fast=True):
    '''
    Dự đoán nhãn cho các đối tượng X_test dựa trên tổ hợp tuyến tính
    của các độ tương đồng có trọng số dựa theo các đối tượng và các đặc trưng.
    
    Đầu vào:
    X_train: scipy.sparse.csr_matrix
        ma trận đặc trưng của tập huấn luyện.
        
    Y_train: scipy.sparse.csr_matrix
        ma trận nhãn của tập huấn luyện.
        
    X_test: scipy.sparse.csr_matrix
        tập kiểm tra. 
    
    k_row: int
        số lân cận gần nhất,
        áp dụng cho tìm kiếm lân cận dựa trên đối tượng.
    
    alpha, beta: float
        hệ số biến đổi hàm mũ.
        
    lamb: float
        hệ số tổ hợp tuyến tính.
                
    Y_ins: scipy.sparse.csr_matrix
        tập dự đoán theo k lân cận dựa vào đối tượng.
    
    Y_fl: scipy.sparse.csr_matrix
        tập dự đoán theo k lân cận dựa vào đặc trưng.
        
    Đầu ra: 
    -------
    Y_lcif: scipy.sparse.csr_matrix
        dự đoán các nhãn cho mỗi đối tượng trong X_test.
    '''
        # chuẩn hóa đối tượng theo dòng
        # dự đoán theo k lân cận gần nhất dựa trên đối tượng.
    X_train_norm_rows = normalize_rows(X_train)
    X_test_norm_rows = normalize_rows(X_test)
    # phi = create_index_partition(X_train_norm_rows, 100)
    
    if knn_fast:
        phi = create_index_partition(X_train_norm_rows, 100)
        kNN_ins = instance_knn_fast(X_test_norm_rows, k_rows, phi,
                                    simil_method, k_power)
    else:
        kNN_ins = instance_knn_search(X_test_norm_rows, k_rows, X_train_norm_rows,
                                      simil_method, k_power)
                                      
    Y_ins = instance_knn_predict(kNN_ins, alpha, Y_train)

    # chuẩn hóa đối tượng theo cột
    # dự đoán theo k lân cận gần nhất dựa trên đặc trưng.
    X_train_norm_ft, norm = normalize_features(X_train)
    X_test_norm_ft, norm = normalize_features(X_test, norm)

    S = create_simil_matrix(X_train_norm_ft, Y_train)
    
    lcard = cardinality(Y_train)
    Y_fl = feature_knn_predict(X_test_norm_ft, S, beta, round(lcard))
    
    # Y_fl = feature_knn_predict(X_test_norm_ft, S, beta)
    
    Y_lcif = Y_ins.multiply(lamb) + Y_fl.multiply(1-lamb)

    if not return_all:
        return Y_lcif
    else:
        return (Y_ins, Y_fl, Y_lcif)
    
def _select_rows_taat(shared_objects, row_indices, col_data):
    I_taat, m = shared_objects
    if row_indices.size > 0:
        # sắp xếp và chọn ra k dòng có giá trị lớn nhất
        # ở cột đang xét
        if col_data.size < m:
            m = col_data.size
        top_m_idx = np.argpartition(col_data, -m)[-m:]

        # ghi lại những dòng được chọn vào tập TAAT
        for idx in row_indices[top_m_idx]:
            I_taat[idx] = True

# @njit(parallel=True, nogil=True)
# def _select_rows_taat(indptr, indices, data, idx_taat, n, m):
#     for i in prange(n):
#         start, end = indptr[i:i+2]
#         row_indices = indices[start:end]
#         col_data = data[start:end]

#         if col_data.size <= m:
#             idx_taat[row_indices] = True
#         else:
#             top_m_id = np.argpartition(col_data, -m)[-m:]
#             idx_taat[top_m_id] = True

def create_index_partition(features, m, debug=False):
    '''
    Hàm phân vùng tập đặc trưng thành 2 phần
    term-at-a-time (TAAT) và document-at-a-time (DAAT).
    
    Những đối tượng có giá trị đặc trưng trong nhóm m giá trị lớn nhất
    được thêm vào tập TAAT
    
    Đầu vào:
    --------
    features: scipy.sparse.csr_matrix
        tập đặc trưng của tập huấn luyện.
        
    m: int
        số giá trị lớn nhất ở mỗi đặc trưng.
        Những dòng thuộc m giá trị lớn nhất ở mỗi cột
        được thêm vào tập TAAT.
        
    Đầu ra:
    -------
    features_taat: scipy.sparse.csr_matrix
        tập đặc trưng TAAT.
        
    features_taat: scipy.sparse.csc_matrix
        tập đặc trưng DAAT.
        
    max_daat: scipy.sparse.csr_matrix
        giá trị lớn nhất ở mỗi cột trong tập DAAT.
    '''
    features_csc = features.tocsc() # tạo ma trận csc để dễ duyệt theo từng cột
    
    # tạo mảng ghi lại những dòng được đưa vào tập TAAT
    # truy cập mảng chỉ mục, dữ liệu và cắt lát cho từng cột
    
    indptr, indices, data = features_csc.indptr, features_csc.indices, features_csc.data

    # I_taat = np.full(features.shape[0], False, np.bool_)
    # _select_rows_taat(indptr, indices, data, I_taat, features.shape[0], m)
    
    if features.shape[1] >= (cpu_count() * 64):
        # chạy song song đa tiến trình trên các đối tượng
        taat_idx_shr = Array(c_bool, features.shape[0], lock=False)
        with memoryview(taat_idx_shr).cast('B').cast('?') as taat_arr_view:
            with WorkerPool(shared_objects=(taat_arr_view, m)) as pool:
                results = pool.map_unordered(_select_rows_taat, ((indices[start:end], data[start:end])
                                                                for (start, end) in zip(indptr, indptr[1:])),
                                                                iterable_len=features.shape[0])
        I_taat = np.frombuffer(taat_idx_shr, np.bool_)
    else:
    # chạy tuần tự từng đối tượng
        I_taat = np.full(features.shape[0], False, np.bool_)
        for start, end in zip(indptr, indptr[1:]):
            if start < end:
                row_indices = indices[start:end]
                col_data = data[start:end]

                # sắp xếp và chọn ra k dòng có giá trị lớn nhất
                # ở cột đang xét
                if col_data.size < m:
                    k = col_data.size
                    top_m_idx = np.argpartition(col_data, -k)[-k:]
                else:
                    top_m_idx = np.argpartition(col_data, -m)[-m:]

                # ghi lại những dòng được chọn vào tập TAAT
                I_taat[row_indices[top_m_idx]] = True
                
    if debug:
        print(f'TAAT: {I_taat.sum()} instances')
#     tạo tập TAAT và DAAT
    
    features_taat = features.copy()
    features_daat = features.copy()
    taat_indptr = features_taat.indptr
    daat_indptr = features_daat.indptr
    for i, row in enumerate(I_taat):
        if row == False:
            features_taat.data[taat_indptr[i]:taat_indptr[i+1]] = 0
        else:
            features_daat.data[daat_indptr[i]:daat_indptr[i+1]] = 0

    features_taat.eliminate_zeros()    
    features_daat.eliminate_zeros()
    features_daat = features_daat.tocsc()
    
    features_daat.sort_indices()
    max_daat = features_daat.max(0).tocsr()
    
    return features_taat, features_daat, max_daat

def _instance_knn_daat(shared_objects, xid, x, knn_row, knn_data, max_w):
    (daat, result_indices, result_simil,
    idx_bounds, k, simil_method, k_power,
    indices_dtype, data_dtype, count) = shared_objects
    offsets = np.zeros(daat.shape[1], np.uint32)
    # idx_bounds = np.diff(daat.indptr)
    
    while True:
        # tìm đối tượng ứng viên trong tập DAAT
        valid_offset = offsets[x.indices] < idx_bounds[x.indices]
        if valid_offset.sum() == 0:
            break
            
        feature_ids = x.indices[valid_offset]
        feature_offset = daat.indptr[feature_ids] + offsets[feature_ids]

        candidates = daat.indices[feature_offset]
        candidates.sort()

        # tìm đối tượng có độ tương đồng thấp nhất hiện tại
        # lấy ra gái trị độ tương đồng.
        if len(knn_data) > 0:
            min_simil_id = np.argmin(knn_data)
            min_simil = knn_data[min_simil_id]
        else:
            min_simil, min_simil_id = 0, -1

        # tìm ứng viên đầu tiên có điểm xét duyệt cao hơn ngưỡng
        # nếu không có thì dừng.
        upper_bound, pivot_id = 0, -1
        # for j in range(candidates.size):
        for i, candidate in enumerate(candidates):
            upper_bound += max_w[0, feature_ids[i]]
            if upper_bound > min_simil:
                pivot_id = candidate
                break

        if pivot_id == -1:
            break

        # tính độ tương đồng của ứng viên với đối tượng truy vấn đang xét.
        # cập nhật tập KNN.
        pivot = daat[pivot_id]
        # pivot_simil = x.dot(pivot.transpose())[0, 0]
        pivot_simil = compute_similarity(x, pivot, simil_method, k_power)[0, 0]
        
        # debug: count number of computation in DAAT traverse
        if count is not None:
            with count.get_lock():
                count.value += 1

        if pivot_simil > min_simil: 
            # kNN[i, pivot_id] = pivot_simil
            knn_row.append(pivot_id)
            knn_data.append(pivot_simil)
            # knn_row.sort()
            # knn_data.insert(knn_row.index(pivot_id), pivot_simil)
            
            if len(knn_data) > k:
                knn_row.pop(min_simil_id)
                knn_data.pop(min_simil_id)

        # bỏ qua các đối tượng có chỉ mục thấp hơn ứng viên được chọn ở trên.
        for idx in x.indices:
            start, end = daat.indptr[idx:idx+2]
            indices = daat.indices[start:end]
            next_offset = np.nonzero(indices > pivot_id)[0]

            if next_offset.size > 0:
                offsets[idx] = indices[next_offset[0]]
            else:
                offsets[idx] = idx_bounds[idx]

    # return knn_row, knn_data
    # for lid, gid in enumerate(range(xid*k, xid*k+k)):
    start = xid * k
    end = start + len(knn_row)
    
    result_indices[start:end] = np.array(knn_row, dtype=indices_dtype, copy=False)
    result_simil[start:end] = np.array(knn_data, dtype=data_dtype, copy=False)
    
def traverse_daat_candidates(Xq, taat_simil, daat, max_w, k, simil_method='cosine', k_power=None, debug=False):
    '''
    
    '''
    # chuyển tập KNN sang dạng 'list of list'
    # để dễ dàng thêm và xóa phần tử ở ô cụ thể
    kNN = taat_simil.tolil()
    
#     tính ngưỡng xét duyệt đối tượng trong tập DAAT
    # max_w = Xq.multiply(upper_bound)
    # max_w.sort_indices()
    # max_w = max_w.todok()
    
    # ft_daat = phi[1]
    # offsets = np.zeros(ft_daat.shape[1], np.uint32)
    idx_bounds = np.diff(daat.indptr)

    sh_arr_size = kNN.shape[0] * k
    indices_dtype = taat_simil.indices.dtype
    data_dtype = taat_simil.data.dtype
    
    result_indices = Array(indices_dtype.char, sh_arr_size, lock=False)
    result_simil = Array(data_dtype.char, sh_arr_size, lock=False)
    
    
    with memoryview(result_indices).cast('B').cast(indices_dtype.char) as indices_view, \
         memoryview(result_simil).cast('b').cast(data_dtype.char) as data_view:
        if not debug:
            count = None
            
        else:
            count = Value('I', 0)
            
        share_objects = (daat, indices_view, data_view,
                    idx_bounds, k, simil_method, k_power,
                    indices_dtype, data_dtype, count)
            
        with WorkerPool(shared_objects=share_objects) as pool:
            results = pool.map_unordered(_instance_knn_daat, ((i, x, knn_row, knn_data, max_wi)
                                                        for (i, (x, knn_row, knn_data, max_wi))
                                                        in enumerate(zip(Xq, kNN.rows, kNN.data, max_w))),
                                                        iterable_len=Xq.shape[0])
    if debug:
        print(f'compute similarity for {count.value} pairs of instances in DAAT and test')
        
    kNN = csr_matrix((result_simil, result_indices, range(0, sh_arr_size+1, k)), shape=kNN.shape)
    kNN.has_sorted_indices = False
    kNN.eliminate_zeros()
    kNN.sort_indices()
    return kNN

def instance_knn_fast(Xq, k, phi, simil_method='cosine', k_power=None, debug=False):
    '''
    Hàm tìm k lân cận gần nhất theo cách duyệt trên 2 tập TAAT vả DAAT
    kết hợp cắt tỉa.
    
    Đầu vào:
    --------
    Xq: scipy.sparse.csr_matrix
        tập đặc trưng của các đối tượng truy vấn.
        
    k: int
        số lân cận gần nhất.
        
    phi: tuple
        bộ gồm các thành phần:
            0. features_taat: scipy.sparse.csr_matrix
                tập term-at-a-time.
                
            1. features_daat: scipy.sparse.csc_matrix
                tập document-at-a-time
                
            2. max_daat: scipy.sparse.csr_matrix
                mảng giá trị lớn nhất cho mỗi cột ở tập DAAT.
                
    Đầu ra:
    -------
    kNN: scipy.sparse.csr_matrix
        k đối tượng huấn luyện có độ tương đồng lớn nhất với đối tượng truy vấn.
    '''
    kNN = instance_knn_search(Xq, k, phi[0], simil_method, k_power)
    
#     kiểm tra tập DAAT, nếu không có đối tượng nào thì kết thúc
    if phi[1].nnz == 0:
        if debug:
            print('compute similarity for 0 pair of instances in DAAT and test')
        return kNN
    else:
        max_w = Xq.multiply(phi[2])
        max_w.sort_indices()
        return traverse_daat_candidates(Xq, kNN, phi[1], max_w, k, simil_method, k_power, debug)

def grid_search_ins_based(X_train, X_test, Y_train, Y_test, metric,
               simil_method='cosine', k_power=None, knn_fast=True):
    best_params = dict.fromkeys(['k_rows', 'alpha'], 0)

    k_range = [1, 5, 50, 100, 150, 200, 250, 300, 350]
    power_range = [.5, 1, 1.5, 2]
    
    # Instance KNN
    # chuẩn hóa dữ liệu theo dòng
    # tính phân vùng term-at-a-time và document-at-a-time
    features_row_norm = normalize_rows(X_train)
    features_val_row_norm = normalize_rows(X_test)
    
    phi = None
    if knn_fast:
        phi = create_index_partition(features_row_norm, 100)
    
    best_result_init = 0
    if metric is hamming_loss:
        best_result_init = np.inf
        
    best_k_rows = 0
    best_result = best_result_init
    best_kNN = None
        
    print('kRows: ', end='')
    for k_rows in k_range:
        print(f'{k_rows} ', end='')
        if knn_fast:
            knn = instance_knn_fast(features_val_row_norm, k_rows, phi,
                                    simil_method, k_power)
        else:
            knn = instance_knn_search(features_val_row_norm, k_rows, features_row_norm,
                                      simil_method, k_power)
        
        Y_pred = instance_knn_predict(knn, 1, Y_train)
        metric_result = evaluate(Y_test, Y_pred, metric)

        if compare_metric_score(best_result, metric_result, metric):
            best_result = metric_result
            best_kNN = knn
            best_k_rows = k_rows
        
    best_params['k_rows'] = best_k_rows
    print()
    return best_params

def grid_search(X_train, X_test, Y_train, Y_test, metric,
               simil_method='cosine', k_power=None, knn_fast=True):
    '''
    Hàm tìm kiếm tham số tốt nhất cho mô hình LCIF.
    Sử dụng kiểm tra chéo 10 Fold để tìm bộ siêu tham số tốt nhất
    Thực hiện tìm kiếm siêu tham số tốt nhất cho từng thành phần:
        - Dự đoán theo k lân cận dựa trên đối tượng: k_rows, alpha.
        - Dự đoán theo k lân cận dựa trên đặc trưng: beta.
        - Tổ hợp tuyến tính 2 phương pháp trên: lambda.
        
    Đầu vào:
    --------
    features, labels: scipy.sparse.csr_matrix
        tập đặc trưng và tập nhãn của các đối tượng huấn luyện.
        
    metric và method tham khảo hàm evaluate.
    
    Đầu ra:
    -------
    best_params: dict
        từ điển gồm các tham số tốt nhất.
    '''
    best_params = dict.fromkeys(['k_rows', 'alpha', 'beta', 'lambda'], 0)

    k_range = [1, 5, 50, 100, 150, 200, 250, 300, 350]
    power_range = [.5, 1, 1.5, 2]
    lambda_range = np.arange(0, 1.1, 0.1)
    
    # Instance KNN
    # chuẩn hóa dữ liệu theo dòng
    # tính phân vùng term-at-a-time và document-at-a-time
    features_row_norm = normalize_rows(X_train)
    features_val_row_norm = normalize_rows(X_test)
    
    phi = None
    if knn_fast:
        phi = create_index_partition(features_row_norm, 100)
    
    best_result_init = 0
    if metric is hamming_loss:
        best_result_init = np.inf
        
    best_k_rows = 0
    best_result = best_result_init
    best_kNN = None
        
    # print('kRows: ', end='')
    for k_rows in k_range:
        # print(f'{k_rows} ', end='')
        if knn_fast:
            knn = instance_knn_fast(features_val_row_norm, k_rows, phi,
                                    simil_method, k_power)
        else:
            knn = instance_knn_search(features_val_row_norm, k_rows, features_row_norm,
                                      simil_method, k_power)
        
        Y_pred = instance_knn_predict(knn, 1, Y_train)
        metric_result = evaluate(Y_test, Y_pred, metric)

        if compare_metric_score(best_result, metric_result, metric):
            best_result = metric_result
            best_kNN = knn
            best_k_rows = k_rows
        
    best_params['k_rows'] = best_k_rows
    # print()

    # tìm kiếm tham số alpha có kết quả độ đo tốt nhất
    best_k_row_power, best_result = -1, best_result_init
    best_Y_ins = None
    
    # print('kRowsPower:', end='')
    for row_power in power_range:
        # print(f'{row_power} ', end='')
        Y_pred = instance_knn_predict(best_kNN, row_power, Y_train)
        metric_result = evaluate(Y_test, Y_pred, metric)
        
        if compare_metric_score(best_result, metric_result, metric):
            best_result = metric_result
            best_k_row_power = row_power
            best_Y_ins = Y_pred
                
    best_params['alpha'] = best_k_row_power
    # print()

    # Feature KNN
    # chuẩn hóa dữ liệu theo cột,
    # tập đánh giá sử dụng giá trị chuẩn hóa từ tập huấn luyện
    train_ft_norm, norm = normalize_features(X_train)
    val_ft_norm, norm = normalize_features(X_test, norm)

    S = create_simil_matrix(train_ft_norm, Y_train)
    lcard = cardinality(Y_train)

    # tìm kiếm tham số beta có kết quả độ đo tốt nhất
    best_k_feature_power, best_result = -1, best_result_init
    best_y_ft = None
    
    # print('featPower:', end='')
    for k_feature_power in power_range:
        # print(f'{k_feature_power} ', end='')
        Y_pred = feature_knn_predict(val_ft_norm, S, k_feature_power, round(lcard))
        # Y_pred = feature_knn_predict(val_ft_norm, S, k_feature_power)
        metric_result = evaluate(Y_test, Y_pred, metric)
        
        if compare_metric_score(best_result, metric_result, metric):
            best_result = metric_result
            best_k_feature_power = k_feature_power
            best_y_ft = Y_pred
                
    best_params['beta'] = best_k_feature_power
    # print()

    # tìm tham số lambda cho tổ hợp tuyến tính
    best_lambda, best_result = -1, best_result_init
    
    # print('lambda', end='')
    for lamb in lambda_range:
        # print(f'{lamb} ', end='')
        # Y_pred = lcif(Y_ins=best_Y_ins, Y_fl=best_y_ft, lamb=lamb)
        Y_pred = best_Y_ins.multiply(lamb) + best_y_ft.multiply(1-lamb)
        
        metric_result = evaluate(Y_test, Y_pred, metric)
        if compare_metric_score(best_result, metric_result, metric):
            best_result = metric_result
            best_lambda = lamb

    best_params['lambda'] = best_lambda
    # print()

    return best_params

def grid_search_fold(features, labels, metric,
                    simil_method='cosine', k_power=None, knn_fast=True):
    '''
    Hàm tìm kiếm tham số tốt nhất cho mô hình LCIF.
    Sử dụng kiểm tra chéo 10 Fold để tìm bộ siêu tham số tốt nhất
    Thực hiện tìm kiếm siêu tham số tốt nhất cho từng thành phần:
        - Dự đoán theo k lân cận dựa trên đối tượng: k_rows, alpha.
        - Dự đoán theo k lân cận dựa trên đặc trưng: beta.
        - Tổ hợp tuyến tính 2 phương pháp trên: lambda.
        
    Đầu vào:
    --------
    features, labels: scipy.sparse.csr_matrix
        tập đặc trưng và tập nhãn của các đối tượng huấn luyện.
        
    metric và method tham khảo hàm evaluate.
    
    Đầu ra:
    -------
    best_params: dict
        từ điển gồm các tham số tốt nhất.
    '''
    best_params = dict.fromkeys(['k_rows', 'alpha', 'beta', 'lambda'], 0)

    n_folds, i = 10, 0
    k_fold = KFold(n_splits=n_folds)
        
    for train_indices, val_indices in k_fold.split(features):
        print(f'fold {i}')
        best_params_fold = grid_search(features[train_indices], features[val_indices], 
                                       labels[train_indices], labels[val_indices], metric,
                                      simil_method, k_power, knn_fast)
        for key in best_params:
            best_params[key] += best_params_fold[key]

        i += 1
            
    for key in best_params:
        best_params[key] /= n_folds
        
    best_params['k_rows'] = round(best_params['k_rows'])
    return best_params

def run_ins_search(X_train, X_test, Y_train, Y_test,
             k_rows, alpha, metric, return_pred=False,
            simil_method='cosine', k_power=None):
    X_train_norm_rows = normalize_rows(X_train)
    X_test_norm_rows = normalize_rows(X_test)

    knn = instance_knn_search(X_test_norm_rows, k_rows, X_train_norm_rows,
                             simil_method, k_power)

    Y_pred = instance_knn_predict(knn, alpha, Y_train)
    
    metric_result = evaluate(Y_test, Y_pred, metric)
    
    if not return_pred:
        return metric_result
    else:
        return Y_pred, metric_result

def run_ins_fast(X_train, X_test, Y_train, Y_test,
             k_rows, alpha, metric, return_pred=False,
            simil_method='cosine', k_power=None, debug=True):
    X_train_norm_rows = normalize_rows(X_train)
    X_test_norm_rows = normalize_rows(X_test)

    phi = create_index_partition(X_train_norm_rows, 100, debug)
    knn = instance_knn_fast(X_test_norm_rows, k_rows, phi,
                                simil_method, k_power, debug)

    Y_pred = instance_knn_predict(knn, alpha, Y_train)
    
    metric_result = evaluate(Y_test, Y_pred, metric)
    
    if not return_pred:
        return metric_result
    else:
        return Y_pred, metric_result

def run_feat_knn(X_train, X_test, Y_train, Y_test, beta, metric,
                return_pred=False, evaluate_all=False):
    X_train_norm_ft, norm = normalize_features(X_train)
    X_test_norm_ft, norm = normalize_features(X_test, norm)

    S = create_simil_matrix(X_train_norm_ft, Y_train)
    
    lcard = cardinality(Y_train)
    Y_pred = feature_knn_predict(X_test_norm_ft, S, beta, round(lcard))

    metric_result = evaluate(Y_test, Y_pred, metric)
    
    if not return_pred:
        return metric_result
    else:
        return Y_pred, metric_result

def run_lcif(X_train, X_test, Y_train, Y_test,
             k_rows, alpha, beta, lamb, metric,
             return_pred=False, evaluate_all=False,
            simil_method='cosine', k_power=None, knn_fast=True):
    '''
    Hàm thực hiện chạy mô hình trên tập dữ liệu và đánh giá trên độ đo cho trước.
    
    Đầu vào:
    --------
    X_train, X_test, Y_train, Y_test: scipy.sparse matrix
        ma trận thưa lần lượt thể hiện:
        - tập đặc trưng huấn luyện
        - tập đặc trưng kiểm tra
        - tập nhãn huấn luyện
        - tập nhãn kiểm tra
        
    k_rows: int
        số lân cận gần nhất cho mô hình thành phần k-NN dựa trên đối tượng
        
    alpha, beta, lamb: float
        hệ số mũ cho mô hình thành phần, lần lượt dựa trên đối tượng và đặc trưng và hệ số kết hợp.
        
    metric: callable
        độ đo đánh giá
        
    return_pred: bool
        nếu True trả về điểm dự đoán các dối tượng kiểm tra.
        
    evaluate_all: bool
        nếu True đánh giá dự đoán của các mô hình thành phần và tổng hợp.
        
    simil_method: str
        chọn phương pháp tính độ tương đồng (hiện tại chỉ áp dụng cho mô hình dựa trên đối tượng).
            - "cosine": độ tương đồng cosine
            - "RES": độ tương đồng cộng hưởng.
            
    k_power: array-like, len(k_power) = 4
        mảng hệ số mũ các thành phần, chỉ dùng khi simil_method == "RES"
        
    knn_fast: bool
        áp dụng phân chia tập huấn luyện và kỹ thuật xử lý và xét duyệt từng dòng - Document-at-a-time.
        
    Đầu ra:
    -------
    metric_result: float
        Kết quả đánh gía trên độ đo.
        
    Y_preds: scipy.sparse.csr_matrix
        ma trận điểm tin cậy dự đoán các đối tượng, trả về khi return_pred == True
    '''
    
    Y_preds = lcif(X_train, Y_train, X_test, k_rows, alpha, beta, lamb,
                   evaluate_all, simil_method, k_power, knn_fast)
    
    if not evaluate_all:
        metric_result = evaluate(Y_test, Y_preds, metric)
    else:
        metric_result = [evaluate(Y_test, Y_pred, metric) for Y_pred in Y_preds]
    
    if not return_pred:
        return metric_result
    else:
        return Y_preds, metric_result

def run_lcif_gs_large(X_train, X_test, Y_train, Y_test, metric,
                      return_params=False, evaluate_all=True,
                      simil_method='cosine', k_power=None, knn_fast=True):
    '''
    Hàm tìm kiếm lưới siêu tham số tốt nhất kết hợp kiểm tra chéo 10 phần
    và áp dụng mô hình với siêu tham số tìm được lên tập dữ liệu.
    
    Đầu vào:
    --------
    return_params: bool
        nếu true trả về bộ siêu tham số tốt nhất tìm được.
        
    Các tham số khác có ý nghĩa tương tự hàm run_lcif.
    
    Đầu ra:
    -------
    params: dict
        bộ siêu tham số tốt nhất tìm được, trả về khi return_params == True.
        
    metric_results: float
        kết quả đánh giá trên độ đo được chọn.
    '''
    compare_n_resize(X_train, X_test)
    compare_n_resize(Y_train, Y_test)
        
    params = grid_search_fold(X_train, Y_train, metric, simil_method, k_power, knn_fast)
    metric_results = run_lcif(X_train, X_test, Y_train, Y_test,
                              params['k_rows'], params['alpha'], params['beta'], params['lambda'],
                              metric, evaluate_all=evaluate_all,
                              simil_method=simil_method, k_power=k_power, knn_fast=knn_fast)
    
    if return_params:
        return params, metric_results
    else:
        return metric_results
    
def compute_entropy_feature(feature_scores, labels, fid):
    '''
    Hàm tính độ thông tin Entropy cho một đặc trưng trong tập dữ liệu.
    Độ thông tin Entropy là tổng các giá trị Entropy các nhãn của những đối tượng có đặc trưng được xét.
    Kết quả lưu vào mảng feature_scores.
    
    Đầu vào:
    --------
    feature_scores: np.ndarray
        mảng lưu kết quả
        
    labels: scipy.sparse.csr_matrix
        ma trận nhãn của các đối tượng có đặc trưng được xét.
        
    fid: int
        chỉ mục của đặc trưng được xét.
    '''
    if labels.indices.size > 0:
        label_counts = np.array(labels.sum(0).reshape(-1), np.float32, copy=False)
        label_counts /= label_counts.shape[1]
        label_counts = np.multiply(label_counts,
                                    np.log2(label_counts, where=label_counts>0))
        
        feature_scores[fid] = label_counts.sum()
    
def compute_entropy_all_features(features, labels):
    '''
    Hàm tính độ thông tin Entropy cho một đặc trưng trong tập dữ liệu.
    
    Đầu vào:
    --------
    features, labels: scipy.sparse.csr_matrix
        ma trận đặc trưng và nhãn của tập dữ liệu.
        
    Đầu ra:
    -------
    feature_scores: np.ndarray
        mảng kết quả Entropy của các đặc trưng.
    '''
    features_csc = features.tocsc()
    indices, indptr = features_csc.indices, features_csc.indptr

    # chạy song song đa tiến trình trên các đặc trưng
    feature_scores = Array('f', features.shape[1], lock=False)
    with memoryview(feature_scores).cast('b').cast('f') as scores_arr_view:
        with WorkerPool(shared_objects=scores_arr_view) as pool:
            pool.map_unordered(compute_entropy_feature, ((labels[indices[start:end]], i)
                                                         for i, (start, end) in enumerate(zip(indptr, indptr[1:]))),
                               iterable_len=indptr.size - 1)

    # chạy tuần tự từng đặc trưng
#     feature_scores = np.zeros(features.shape[1])

#     for i, (start, end) in enumerate(zip(indptr, indptr[1:])):
#         if start < end:
#             rows = features_csc.indices[start:end]
            
#             label_counts = np.array(labels[rows].sum(0).reshape(-1), np.float32, copy=False)
#             label_counts /= rows.size
#             label_counts = np.multiply(label_counts,
#                                        np.log2(label_counts, where=label_counts>0), out=label_counts)
            
#             feature_scores[i] = label_counts.sum()
        
    return feature_scores

def search_best_representation(X_train, X_test, Y_train, Y_test, feature_scores, metric,
                              simil_method='cosine', k_power=None, knn_fast=True):
    '''
    Hàm tìm số đặc trưng tốt nhất của tập dữ liệu dựa vào giá trị Entropy của các cột đặc trưng.
    Lựa chọn số đặc trưng tốt nhất theo thứ tự giảm dần của giá trị Entropy các đặc trưng.
    
    Đầu vào:
    --------
    X_train, X_test, Y_train, Y_test: scipy.sparse matrix
        ma trận thưa lần lượt thể hiện:
        - tập đặc trưng huấn luyện
        - tập đặc trưng kiểm tra
        - tập nhãn huấn luyện
        - tập nhãn kiểm tra
        
    feature_scores: np.ndarray
        mảng giá trị Entropy của các đặc trưng
        
    metric: callable
        độ đo đánh giá
        
    simil_method: str
        chọn phương pháp tính độ tương đồng (hiện tại chỉ áp dụng cho mô hình dựa trên đối tượng).
            - "cosine": độ tương đồng cosine
            - "RES": độ tương đồng cộng hưởng.
            
    k_power: array-like, len(k_power) = 4
        mảng hệ số mũ các thành phần, chỉ dùng khi simil_method == "RES"
        
    knn_fast: bool
        áp dụng phân chia tập huấn luyện và kỹ thuật xử lý và xét duyệt từng dòng - Document-at-a-time.
        
    Đầu ra:
    -------
    max_k_LCIF: int
        số đặc trưng tốt nhất có thể giữ lại mà vẫn lưu trữ được nhiều thông tin nhất.
    '''
    
    k_feature_selection_range = [0.01, 0.1, 0.25, 0.5, 0.75, 0.99, 1.0]
    k_rows = 100; k_row_power = 1; k_feature_power = 1; fixed_lambda = 0.5
    max_score_LCIF = -1; max_k_LCIF=-1
    
    X_train_csc = X_train.tocsc()
    X_test_csc = X_test.tocsc()
    for k_percent in k_feature_selection_range:
        k_features = round(X_train.shape[1] * k_percent)
        top_k_features = np.argpartition(feature_scores, -k_features)[-k_features:]
        top_k_features.sort()
        
        X_train_top_k = X_train_csc[:, top_k_features]
        X_test_top_k = X_test_csc[:, top_k_features]
        
        results = run_lcif(X_train_top_k, X_test_top_k, Y_train, Y_test,
                           k_rows, k_row_power, k_feature_power, fixed_lambda, metric,
                          simil_method=simil_method, k_power=k_power)
        
            
        # if results > max_score_LCIF:
        if compare_metric_score(max_score_LCIF, results, metric):
            max_score_LCIF = results
            max_k_LCIF = k_features
            
    return max_k_LCIF

def run_lcif_gs_extreme(X_train, X_test, Y_train, Y_test,
                        metric, return_params=True, evaluate_all=True,
                       simil_method='cosine', k_power=None, knn_fast=True):
    '''
    Hàm tìm kiếm lưới siêu tham số tốt nhất và áp dụng mô hình lên tập dữ liệu.
    Thực hiện lấy mẫu 1000/10000 đối tượng đầu tiên của tập kiểm tra làm tập đánh giá
    cho việc tìm kiếm siêu tham số.
    
    Các tham số và kết quả trả về tương tự với hàm run_lcif_gs_large.
    '''
    # xác định số đối tượng lấy mẫu
    sample_test = 100
    if X_test.shape[0] > 10000:
        sample_test = 10000
    elif X_test.shape[0] > 1000:
        sample_test = 1000

    # tính giá trị Entropy cho các cột đặc trưng
    feature_scores = compute_entropy_all_features(X_train, Y_train)

    X_test_samples = X_test[:sample_test]
    Y_test_samples = Y_test[:sample_test]

    # tìm kiếm không gian đặc trưng mới biểu diễn dữ liệu tốt nhất
    optimal_k_select = search_best_representation(X_train, X_test_samples, Y_train, Y_test_samples,
                                                  feature_scores, metric, simil_method, k_power, knn_fast)

    top_k_features = np.argpartition(feature_scores, -optimal_k_select)[-optimal_k_select:]
    top_k_features.sort()

    X_train_top_k_ft = X_train.tocsc()[:, top_k_features]
    X_test_samples_top_k_ft = X_test_samples.tocsc()[:, top_k_features]
    
    params = grid_search(X_train_top_k_ft, X_test_samples_top_k_ft, Y_train, Y_test_samples, metric,
                        simil_method, k_power, knn_fast)

    X_test_top_k_ft = X_test.tocsc()[:, top_k_features]
    metric_results = run_lcif(X_train_top_k_ft, X_test_top_k_ft, Y_train, Y_test,
                                        params['k_rows'], params['alpha'], params['beta'], params['lambda'],
                                        metric, evaluate_all=evaluate_all,
                                        simil_method=simil_method, k_power=k_power, knn_fast=knn_fast)
    
    if return_params:
        return params, metric_results
    else:
        return metric_results
