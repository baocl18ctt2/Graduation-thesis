import numpy as np
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix, dok_matrix
from mpire import WorkerPool
from multiprocessing.sharedctypes import Array
from os import cpu_count

GS_BIG = False
PARALLEL_THRESHOLD = 1024

def cardinality(spmatrix):
    '''
    Hàm tính trung bình số cột nhận giá trị khác 0 ở mỗi dòng của ma trận.
    Lưu ý: ma trận cần hỗ trợ phương thức khử giá trị 0.
    
    Đầu vào:
    --------
    spmatrix: lớp đối tượng ở module scipy.sparse
        tập dữ liệu.
    
    Đầu ra:
    -------
    card: float
        số lượng cột có giá trị != 0 trung bình ở mỗi dòng.
    '''
#     original_data = spmatrix.data.copy()
    
#     spmatrix.eliminate_zeros()
#     spmatrix.data = spmatrix.data / spmatrix.data    
    
#     mean_each_instance = spmatrix.sum(1).mean()
#     spmatrix.data[:] = original_data[:]
#     return mean_each_instance

    indicator = get_indicator(spmatrix)
    return indicator.sum(1).mean()

# Một số tập dữ liệu đi kèm với mã nguồn của bài báo
# không hoàn toàn chính xác theo định dạng libsvm
# nên cần đọc từng dòng để lấy ra các giá trị đặc trưng và nhãn.
def parse_lines(file_obj, n_instances, n_features=None, n_labels=None):
    '''
    Đọc tập dữ liệu, ghi lại các đặc trưng và nhãn của các đối tượng.
    Tạo 2 ma trận thưa định dạng csr_matrix lưu các đặc trưng và nhãn của các đối tượng.
    
    Đầu vào:
    --------
    file_obj: file object
        đối tượng tập tin tập dữ liệu.
    n_instances: int
        số lượng đối tượng trong tập dữ liệu.
    n_features, n_labels: int, mặc định = None
        số lượng đặc trưng/nhãn trong tập dữ liệu. Nếu là None thì sẽ suy ra từ tập dữ liệu.
        Ngược lại lấy giá trị lớn nhất giữa giá trị truyền vào và đọc được từ tập dữ liệu.
            
    Đầu ra:
    -------
        - features: scipy.sparse.csr_matrix
            ma trận thưa lưu đặc trưng các đối tượng.
        -labels: scipy.sparse.csr_matrix
            ma trận thưa lưu nhãn của các đối tượng.
    '''
    # khởi tạo danh sách chứa các chỉ mục theo dòng, cột
    # và giá trị các đặc trưng và nhãn.
    rows_labels = []; cols_labels = []; labels = []
    rows_features = []; cols_features = []; features = []

    # i = 0
    # while i < n_instances:
    for i in range(n_instances):
        line = file_obj.readline().decode('utf-8')
        
#         xác định dấu cách phần nhãn và đặc trưng.
        first_feature = line.find(':')
        feature_start = line[:first_feature].rfind(' ')

#         thêm các chỉ mục và giá trị
        col_labels = line[:feature_start].split(',')
        if col_labels[0] != '':
            cols_labels.extend(list(map(int, col_labels)))
            rows_labels.extend([i] * len(col_labels))
            labels.extend([1] * len(col_labels))

        for feature_value in line[feature_start + 1:].split():
            feat, val = feature_value.split(':')

            rows_features.append(i)
            cols_features.append(int(feat))
            features.append(float(val))
            
    # suy ra số lượng đặc trưng và nhãn từ các chỉ mục.
    # so sánh mvới giá trị truyền vào (nếu có),
    # chọn giá trị lớn nhất. 
    n_f = max(cols_features) + 1
    if (n_features is None) or (n_features < n_f):
        n_features = n_f
        
    min_label_id = min(cols_labels)
    if min_label_id > 0:
        cols_labels = [label_id - min_label_id for label_id in cols_labels]
        
    n_l = max(cols_labels) + 1
    if (n_labels is None) or (n_labels < n_l):
        n_labels = n_l

    features = csr_matrix((features, (rows_features, cols_features)), (n_instances, n_features))
    labels = csr_matrix((labels, (rows_labels, cols_labels)), (n_instances, n_labels))
    
    return features, labels

def list_to_sparse(ls_tup, n_instances, n_labels):
    '''
    Chuyển đổi danh sách các bộ nhãn của các đối tượng qua dạng ma trận thưa.
    Danh sách trên là kết quả đọc dữ liệu sử dụng hàm load_svmlight_file của module sklearn.dataset.
    
    Đầu vào:
    --------
    ls_tup: list
        danh sách các bộ nhãn của các đối tượng.
    n_instances: int
        số lượng đối tượng.
    n_labels: int
        số nhãn. 
        
    Đầu ra: ma trận thưa lưu các nhãn của các đối tượng, kích thước n_instances x n_labels.
    '''
    row, col = [], []
    for i, tup in enumerate(ls_tup):
        col.extend(tup)
        row.extend([i]*len(tup))
        
    max_label_id = max(col)
    if n_labels <= max_label_id:
        n_labels = int(max_label_id + 1)
    
    labels = dok_matrix((n_instances, n_labels))
    labels[row, col] = 1
    return labels.tocsr()

def load_data(filepath, n_features=None, n_labels=None):
    '''
    Đọc tập dữ liệu định dạng libsvm và tạo 2 ma trận thưa lưu các đặc trưng và nhãn của các đối tượng.
    
    Nếu các dòng hoàn toàn được ghi chính xác theo định dạng
        <nhãn 1>,...<nhãn n> <đặc trưng 1>:<giá trị 1>  ... <đặc trưng 1>:<giá trị 1> ... <đặc trưng m>:<giá trị m>
    thì sử dụng hàm load_svmlight_file của module sklearn.dataset.
    
    Đầu vào:
    --------
    filepath: chuỗi
        đường dẫn đến tập tin dữ liệu.
    n_features, n_labels: int, mặc định = None
        số lượng đặc trưng/nhãn trong tập dữ liệu. Nếu là None thì sẽ suy ra từ tập dữ liệu.
        Ngược lại lấy giá trị lớn nhất giữa giá trị truyền vào và đọc được từ tập dữ liệu.
            
    Đầu ra:
    -------
    features: ma trận thưa có kích thước (n_instances, n_features)
        ma trận đặc trưng của các đối tượng.
        
    labels: ma trận thưa có kích thước (n_instances, n_labels)
        ma trận nhãn của các đối tượng.
    '''
    # datafile = open(filepath, 'rb')
    with open(filepath, 'rb') as datafile:
        header = datafile.readline().decode('utf-8')
        data_start_pos = datafile.tell()
        n_instances, n_f, n_l = list(map(int, header.split()))

        if n_features is None:
            n_features = n_f
        else:
            n_features = max(n_f, n_features)

        if n_labels is None:
            n_labels = n_l
        else:
            n_labels = max(n_l, n_labels)

        try:
            features, labels = load_svmlight_file(datafile, n_features=n_features, multilabel=True)
            labels = list_to_sparse(labels, n_instances, n_labels) 
        except ValueError:
            datafile.seek(data_start_pos, 0)
            features, labels = parse_lines(datafile, n_instances, n_features, n_labels)

    return features, labels

def normalize_rows(spmatrix):
    norm = np.sqrt(spmatrix.power(2).sum(1))
    inv_norm = np.divide(1., norm, where=norm!=0)
    
    return spmatrix.multiply(inv_norm).tocsr()

def _row_sort_top_k(shared_objects, start, indices, data):
    dst_data, k = shared_objects
    
    if indices.size > 0:
        if data.size < k:
            top_k_idx = np.argpartition(data, -data.size)[-data.size:]
        else:
            top_k_idx = np.argpartition(data, -k)[-k:]

        remove = ~np.isin(indices, indices[top_k_idx], True)
        if remove.size > 0:
            rm_indices = np.nonzero(remove)[0] + start
            for idx in rm_indices: 
                dst_data[idx] = 0

def partial_sort_top_k(spmatrix, k):
    '''
    Hàm lọc ra k cột có giá trị lớn nhất theo từng dòng.
    
    Đầu vào:
    --------
    spmatrix: scipy.sparse.
        tập dữ liệu.
        
    k: int
        số giá trị lớn nhất.
    
    Đầu ra:
    --------
    kNN: scipy.sparse.csr_matrix
        k đối tượng huấn luyện có độ tương đồng lớn nhất với đối tượng truy vấn.
    '''
    # chuyển đổi ma trận đầu vào về kiểu csr_matrix
    # để truy xuất từng dòng hiệu quả.
    spmatrix = csr_matrix(spmatrix)
        
    # truy xuất mảng chỉ mục cột, dữ liệu và lát cắt ứng với từng dòng.
    sp_indices, sp_indptr, sp_data = spmatrix.indices, spmatrix.indptr, spmatrix.data
    nrows, ncols = spmatrix.shape
    
    # truy xuất mảng chỉ mục cột, dữ liệu và lát cắt ứng với từng dòng.
    sp_indices, sp_indptr, sp_data = spmatrix.indices, spmatrix.indptr, spmatrix.data
    
    # kNN = dok_matrix(spmatrix.shape)

    if GS_BIG:
        for i, (start, end) in enumerate(zip(sp_indptr, sp_indptr[1:])):
            # lấy lát cắt mảng chỉ mục và dữ liệu cho dòng i
            # start, end = sp_indptr[i:i+2]
            if start < end:
                indices = sp_indices[start:end]
                data = sp_data[start:end]

                # tìm k cột có giá trị lớn nhất ở dòng i
                # nếu số cột có giá trị < k thì nhận hết các cột
                if data.size < k:
                    top_k_idx = np.argpartition(data, -data.size)[-data.size:]
                else:
                    top_k_idx = np.argpartition(data, -k)[-k:]
                
                rm_indices = ~np.isin(indices, top_k_idx)
                data[rm_indices] = 0
                # kNN[i, indices[top_k_idx]] = data[top_k_idx]
                
        kNN = spmatrix
        kNN.eliminate_zeros()

    # chạy song song đa tiến trình
    elif nrows >= PARALLEL_THRESHOLD:

    #if nrows >= 4096:
        c_type = sp_data.dtype

        dst_data = Array(c_type.char, sp_data.size, lock=False)
        with memoryview(dst_data).cast('b').cast(c_type.char) as dst_data_view:
            dst_data_view[:] = sp_data

            with WorkerPool(shared_objects=(dst_data_view, k)) as pool:
                results = pool.map_unordered(_row_sort_top_k, ((start, sp_indices[start:end], sp_data[start:end])
                                                               for (start, end) in zip(sp_indptr, sp_indptr[1:])),
                                             iterable_len=nrows)

            # spmatrix.data[:] = dst_data_view
            kNN = csr_matrix((dst_data_view, sp_indices.copy(), sp_indptr.copy()), shape=(nrows, ncols))
            kNN.eliminate_zeros()
    
    # chạy tuần tự từng đối tượng
    else:
        kNN = dok_matrix(spmatrix.shape)

        for i, (start, end) in enumerate(zip(sp_indptr, sp_indptr[1:])):
            # lấy lát cắt mảng chỉ mục và dữ liệu cho dòng i
            # start, end = sp_indptr[i:i+2]
            if start < end:
                indices = sp_indices[start:end]
                data = sp_data[start:end]

                # tìm k cột có giá trị lớn nhất ở dòng i
                # nếu số cột có giá trị < k thì nhận hết các cột
                if data.size < k:
                    top_k_idx = np.argpartition(data, -data.size)[-data.size:]
                else:
                    top_k_idx = np.argpartition(data, -k)[-k:]

    
                kNN[i, indices[top_k_idx]] = data[top_k_idx]

    # spmatrix.eliminate_zeros()
        kNN = kNN.tocsr()
        
    kNN.sort_indices()
    return kNN

# def partial_sort_top_k(spmatrix, k):
#     '''
#     Hàm lọc ra k cột có giá trị lớn nhất theo từng dòng.
    
#     Đầu vào:
#     --------
#     spmatrix: scipy.sparse.
#         tập dữ liệu.
        
#     k: int
#         số giá trị lớn nhất.
    
#     Đầu ra:
#     --------
#     kNN: scipy.sparse.csr_matrix
#         k đối tượng huấn luyện có độ tương đồng lớn nhất với đối tượng truy vấn.
#     '''
#     # chuyển đổi ma trận đầu vào về kiểu csr_matrix
#     # để truy xuất từng dòng hiệu quả.
#     spmatrix = csr_matrix(spmatrix)
#     k = int(k)
        
    # truy xuất mảng chỉ mục cột, dữ liệu và lát cắt ứng với từng dòng.
    # sp_indices, sp_indptr, sp_data = spmatrix.indices, spmatrix.indptr, spmatrix.data
    # topk_indices, topk_indptr, topk_data = [], [0], []
    # kNN = dok_matrix(spmatrix.shape)
    # nrows, ncols = spmatrix.shape
    
    # for i, (start, end) in enumerate(zip(sp_indptr, sp_indptr[1:])):
        # # lấy lát cắt mảng chỉ mục và dữ liệu cho dòng i
        # # start, end = sp_indptr[i:i+2]
        # if start < end:
            # indices = sp_indices[start:end]
            # data = sp_data[start:end]

            # # tìm k cột có giá trị lớn nhất ở dòng i
            # # nếu số cột có giá trị < k thì nhận hết các cột
            # if data.size < k:
                # top_k_idx = np.argpartition(data, -data.size)[-data.size:]
            # else:
                # top_k_idx = np.argpartition(data, -k)[-k:]
            

            # kNN[i, indices[top_k_idx]] = data[top_k_idx]
    
    # chạy tuần tự từng đối tượng
    #else:
        #kNN = dok_matrix(spmatrix.shape)


        #for i, (start, end) in enumerate(zip(sp_indptr, sp_indptr[1:])):
            #lấy lát cắt mảng chỉ mục và dữ liệu cho dòng i
            #start, end = sp_indptr[i:i+2]
            #if start < end:
                #indices = sp_indices[start:end]
                #data = sp_data[start:end]


                #tìm k cột có giá trị lớn nhất ở dòng i
                #nếu số cột có giá trị < k thì nhận hết các cột
                #if data.size < k:
                    #top_k_idx = np.argpartition(data, -data.size)[-data.size:]
                #else:
                    #top_k_idx = np.argpartition(data, -k)[-k:]

    
                #kNN[i, indices[top_k_idx]] = data[top_k_idx]

    #spmatrix.eliminate_zeros()
        #kNN = kNN.tocsr()
        
    #kNN.sort_indices()
    #return kNN

# def partial_sort_top_k(spmatrix, k):
#     '''
#     Hàm lọc ra k cột có giá trị lớn nhất theo từng dòng.
    
#     Đầu vào:
#     --------
#     spmatrix: scipy.sparse.
#         tập dữ liệu.
        
#     k: int
#         số giá trị lớn nhất.
    
#     Đầu ra:
#     --------
#     kNN: scipy.sparse.csr_matrix
#         k đối tượng huấn luyện có độ tương đồng lớn nhất với đối tượng truy vấn.
#     '''
#     # chuyển đổi ma trận đầu vào về kiểu csr_matrix
#     # để truy xuất từng dòng hiệu quả.
#     spmatrix = csr_matrix(spmatrix)
#     k = int(k)
        
#     # truy xuất mảng chỉ mục cột, dữ liệu và lát cắt ứng với từng dòng.
#     sp_indices, sp_indptr, sp_data = spmatrix.indices, spmatrix.indptr, spmatrix.data
#     # topk_indices, topk_indptr, topk_data = [], [0], []
#     kNN = dok_matrix(spmatrix.shape)
#     # nrows, ncols = spmatrix.shape
    
#     for i, (start, end) in enumerate(zip(sp_indptr, sp_indptr[1:])):
#         # lấy lát cắt mảng chỉ mục và dữ liệu cho dòng i
#         # start, end = sp_indptr[i:i+2]
#         if start < end:
#             indices = sp_indices[start:end]
#             data = sp_data[start:end]

#             # tìm k cột có giá trị lớn nhất ở dòng i
#             # nếu số cột có giá trị < k thì nhận hết các cột
#             if data.size < k:
#                 top_k_idx = np.argpartition(data, -data.size)[-data.size:]
#             else:
#                 top_k_idx = np.argpartition(data, -k)[-k:]
            
#         #     topk_indices.extend(indices[top_k_idx])
#         #     topk_data.extend(data[top_k_idx])
#         #     topk_indptr.append(topk_indptr[i] + top_k_idx.size)
#         # else:
#         #     topk_indptr.append(topk_indptr[i])
#             kNN[i, indices[top_k_idx]] = data[top_k_idx]

        
#     # kNN = csr_matrix((topk_data, topk_indices, topk_indptr), shape=(nrows, ncols))
#     kNN = kNN.tocsr()
#     kNN.sort_indices()
#     return kNN
    
def normalize_features(features, norm=None):
    if norm is None:
        sum_of_squares = np.sqrt(features.power(2).sum(0))
        norm = np.divide(1., sum_of_squares, where=sum_of_squares!=0)
        
    return features.multiply(norm).tocsr(), norm

def get_indicator(S):
    '''
    Hàm tạo chỉ báo những ô có giá trị khác 0 trong ma trận S.
    
    Tạo ma trận cùng kích thước với S, trong đó những ô ở S
    có giá trị khác 0 nhận giá trị 1, ngược lại 0.
    
    Đầu vào:
    ---------
    - S: lớp đối tượng ma trận thuộc scipy.sparse
    
    Đầu ra:
    ---------
    - indicator: scipy.sparse.csr_matrix
        ma trận đánh dấu những ô có giá trị.
    '''
#     indicator_data = np.zeros_like(S.data, np.uint8)
#     indicator_data[S.data != 0] = 1
    
#     indicator = csr_matrix((indicator_data, S.indices.copy(), S.indptr.copy()),
#                             shape=S.shape)
    
#     indicator.eliminate_zeros()
    indicator = S.copy()
    indicator.eliminate_zeros()
    indicator.data = np.divide(indicator.data, indicator.data,
                               out=indicator.data, casting='unsafe')
    return indicator

def compare_n_resize(mat_a, mat_b):
    if mat_a.shape[1] > mat_b.shape[1]:
        mat_b.resize((mat_b.shape[0], mat_a.shape[1]))
    elif mat_a.shape[1] < mat_b.shape[1]:
        mat_a.resize((mat_a.shape[0], mat_b.shape[1]))
