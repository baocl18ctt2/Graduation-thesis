'''
Tham khảo:
Tan, Zhenhua, and Liangliang He.
"An efficient similarity measure
for user-based collaborative filtering recommender systems
inspired by the physical resonance principle."
IEEE Access 5 (2017): 27211-27228.
'''

import numpy as np
from scipy.sparse import csr_matrix
from numba import njit, prange
from numba.typed import List
from numba import config

# config.THREADING_LAYER = 'tbb'

#TEMPDIR = '/home/u133169/thesis/temp'
#TEMPDIR = '/content/temp'
TEMPDIR = '/home/quytran/Documents/temp'

# mục III.A của bài báo tham khảo
@njit
def initial_phase(val, overal_max, overal_min, overal_med, ins_mean, overal_mean):
    '''
    Tính góc của véc-tơ dữ liệu
    theo giá trị đặc trưng đang xét và trung bình đặc trưng.
    
    Đầu vào:
    --------
    val, overal_max, overal_min, overal_med, ins_mean, overal_mean: float
        lần lượt là:
        - giá trị đặc trưng đang xét.
        - giá trị đặc trưng lớn nhất trên tập dữ liệu.
        - giá trị đặc trưng nhỏ nhất trên tập dữ liệu.
        - giá trị đặc trưng trung vị của tập dữ liệu.
        - giá trị đặc trưng trung bình của đối tượng.
        - giá trị đặc trưng trung bình của tập dữ liệu.
        
    Đầu ra:
    -------
    phi: float
        
    '''
    phi_base = val - overal_med
    per_dist = ins_mean - overal_mean
    cond = phi_base * per_dist

    map_const = np.pi / (overal_max - overal_min)
    if cond >= 0:
        phi = map_const * (1 / abs(per_dist)) * phi_base
    else:
        phi = map_const * (1 + (abs(per_dist) / overal_med)) * phi_base

    return phi

@njit
def consistency(phi_u, phi_v, k1):
    return np.sqrt(0.5 + 0.5 * np.cos(phi_u - phi_v)) ** k1

@njit
def distance(val_u, val_v, feat_mean, k2, k3):
    '''
    Tính hệ số thể hiện tương quan giữa
    1. chênh lệch giá trị đặc trưng của hai đối tượng.
    2. tổng chênh lệch giá trị đặc trưng 2 đối tượng so với trung bình đặc trưng
    
    Nếu 2 giá trị đặc trưng cùng trên/dưới trung bình thì tính 2 thành phần trên,
    ngược lại chỉ tính thành phần (1).
    
    Đầu vào:
    --------
    val_u, val_v: float
        giá trị đặc trưng của 2 đối tượng.
        
    feat_mean: float
        giá trị trung bình đặc trưng.
        
    k2, k3; float
        hệ số mũ.
    
    Đầu ra:
    -------
    dist: float
        hệ số khoảng cách giữa 2 đối tượng.
    '''
    u_diff = val_u - feat_mean
    v_diff = val_v - feat_mean
    cond = u_diff * v_diff

    if cond >= 0:
        dist_pos_1 = np.exp(-abs(val_u - val_v))
        dist_pos_2 = np.exp(0.5 * (abs(u_diff) + abs(v_diff)))
        dist = (dist_pos_1 * dist_pos_2) ** k2
    else:
        dist_pos_1 = np.exp(-abs(val_u - val_v))
        dist = dist_pos_1 ** k3

    return dist

@njit
def jaccard(u_indices, v_indices, k4):
    '''
    Hàm tính hệ số Jaccard thể hiện tỷ lệ phần giao và hợp của các đặc trưng giữa 2 đối tượng.
    
    Đầu vào:
    --------
    u_indices, v_indices: np.ndarray
        mảng chỉ mục đặc trưng của 2 đối tượng.
        
    k4: float
        hệ số mũ.
        
    Đầu ra:
    -------
    jaccard: float
        hệ số Jaccard
    '''
    intersect = np.intersect1d(u_indices, v_indices).size
    # union = np.union1d(u_indices, v_indices).size
    union = u_indices.size + v_indices.size - intersect

    return (intersect / union) ** k4

@njit(parallel=True, nogil=True)
def resonance_similarity(X_shape, X_indices, X_indptr, X_data,
                        Xq_shape, Xq_indices, Xq_indptr, Xq_data,k_arr,
                        feat_mean, overal_max, overal_min, overal_med, overal_mean,
                        simil_indices, simil_data):
    '''
    Hàm tính độ tương đồng cộng hưởng giữa các đối tượng huấn luyện X và truy vấn Xq.
    
    Đầu vào:
    --------
    X_shape, X_indices, X_indptr, X_data: np.ndarray
        lần lượt là mảng kích thước, chỉ mục đặc trưng, phân đoạn dữ liệu cho từng dòng
        và giá trị tại các chỉ mục tương ứng cho các đối tượng huấn luyện.
        
    Xq_shape, Xq_indices, Xq_indptr, Xq_data: np.ndarray
        tương tự như trên cho các đối tượng truy vấn.
        
    feat_mean, overal_max, overal_min, overal_med, overal_mean: float
        lần lượt là:
        - giá trị đặc trưng trung bình của từng đặc trưng.
        - giá trị đặc trưng lớn nhất trên tập dữ liệu.
        - giá trị đặc trưng nhỏ nhất trên tập dữ liệu.
        - giá trị đặc trưng trung vị của tập dữ liệu.
        - giá trị đặc trưng trung bình của tập dữ liệu.
        
    simil_indices, simil_data: numba.typed.List
        danh sách với kiểu dữ liệu xác định đồng nhất các chỉ mục và giá trị cho ma trận tương đồng.
        
    Đầu ra: kết quả được cập nhật vào 2 danh sách simil_indices, simil_data.
    '''
    
    # duyệt từng đối tượng truy vấn
    for i in prange(Xq_shape[0]):
        # xác định phân vùng chỉ mục và giá trị đặc trưng
        xq_start, xq_end = Xq_indptr[i:i+2]
        xq_data = Xq_data[xq_start:xq_end]
        xq_indices = Xq_indices[xq_start:xq_end]
        xq_mean = np.mean(xq_data) # tính trung bình dữ liệu

        # for j, (x_start, x_end) in enumerate(zip(X_indptr, X_indptr[1:])):
        # duyệt từng đối tượng huấn luyện
        for j in prange(X_shape[0]):
            x_start, x_end = X_indptr[j:j+2]
            simil = 0
            x_data = X_data[x_start:x_end]
            x_indices = X_indices[x_start:x_end]
            x_mean = np.mean(x_data)

            # xác định đặc trưng chung
            intersect1d = np.intersect1d(x_indices, xq_indices)
            if intersect1d.size == 0:
                continue
            
            comm1 = np.array([item in intersect1d for item in x_indices]).nonzero()[0]
            comm2 = np.array([item in intersect1d for item in xq_indices]).nonzero()[0]
            
            # tính các giá trị thành phần
            jac = jaccard(x_indices, xq_indices, k_arr[3])

            for comm_id, id1, id2 in zip(intersect1d, comm1, comm2):
                val1, val2 = x_data[id1], xq_data[id2]
                phi_x = initial_phase(val1,overal_max,overal_min, overal_med, x_mean, overal_mean)
                phi_xq = initial_phase(val2, overal_max,overal_min, overal_med, xq_mean, overal_mean)

                cons = consistency(phi_x, phi_xq, k_arr[0])
                dist = distance(val1, val2, feat_mean[comm_id], k_arr[1], k_arr[2])

                simil += cons * dist * jac
            
            # kiểm tra và cập nhật giá trị tương đồng
            if simil > 0:
                simil_indices[i].append(j)
                simil_data[i].append(simil)

@njit
def _create_numba_lil(n):
    '''
    Hàm hỗ trợ khởi tạo danh sách xác định kiểu dữ liệu.
    Khởi tạo 2 danh sách chỉ mục và giá trị cho ma trận tương đồng.
    
    Đầu vào:
    --------
    n: int
        số đối tượng (dòng)
        
    Đầu ra:
    indices, data: numba.typed.List
        danh sách của các danh sách chỉ mục và giá trị.
    '''
    indices = List()
    data = List()

    for i in range(n):
        indices.append(List([0]))
        data.append(List([0.]))

        indices[i].pop()
        data[i].pop()

    return indices, data

def RES(X, Xq, K):
    '''
    Hàm bao ngoài tính độ tương đồng cộng hưởng.
    Lấy ra những mảng dữ liệu phù hợp cho hàm resonance_similarity.
    Tạo ma trận từ danh sách kết quả.
    
    Đầu vào:
    --------
    X, Xq: scipy.sparse.csr_matrix
        ma trận đặc trưng tập huấn luyện và kiểm tra.
        
    K: np.ndarray
        mảng hệ số mũ cho các thành phần độ tương đồng.
        
    Đầu ra:
    -------
    similarity: scipy.sparse.csr_matrix
        ma trận dộ tương đồng giữa các đối tượng kiểm tra và huấn luyện.
    '''    
    X_shape = np.array(X.shape, np.int32, copy=False)
    Xq_shape = np.array(Xq.shape, np.int32, copy=False)
    
    simil_indices, simil_data = _create_numba_lil(Xq_shape[0])

    feat_mean = np.ravel(X.mean(0)) # tính giá trị trung bình từng đặc trưng
    
    # tính các giá trị lớn, nhỏ nhất và trung bình trên tập huấn luyện.
    overal_max, overal_min = X.max(), X.min()
    overal_med = (overal_max - overal_min) / 2
    overal_mean = X.mean()

    resonance_similarity(X_shape, X.indices, X.indptr, X.data,
                        Xq_shape, Xq.indices, Xq.indptr, Xq.data,
                        K, feat_mean, overal_max, overal_min, overal_med, overal_mean,
                        simil_indices, simil_data)

    indptr = np.cumsum([0] + [len(row) for row in simil_indices]) # tính phân đoạn mảng chỉ mục và giá trị tương đồng.
    
    # nối các danh sách thành mảng một chiều
    simil_indices = np.concatenate(simil_indices, axis=None)
    simil_data = np.concatenate(simil_data, axis=None)

    return csr_matrix((simil_data, simil_indices, indptr), (Xq_shape[0], X.shape[0]))
