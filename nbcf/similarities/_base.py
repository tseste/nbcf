import csv
import math
import scipy
import numpy as np
import pandas as pd
import scipy.sparse as sp

from collections import deque
from scipy.sparse.linalg import norm as sparse_norm

from nbcf._utilities import timer
from nbcf._utilities import BaseMultiprocessing


class _CommonMatrixSim(object):
    _centering = None

    def __init__(self, data, user_based):
        self.data = data
        self.user_based = user_based
        self._map_to_array_index()
        rows, columns = self.data[['user', 'item']].nunique().values
        self.train = sp.lil_matrix((rows, columns))
        self._mean_centering()
        iter_data = zip(self.data['user'].values,
                        self.data['item'].values,
                        self.data.rating.values)
        for u, i, r in iter_data:
            self.train[u, i] = r
        del self.data
        self.train = self.train.tocsr()
        if not self.user_based:
            self.train = self.train.T

    def _map_to_array_index(self):
        unique_users = self.data.user.unique()
        unique_users.sort()
        user_sequence = range(self.data.user.nunique())
        self._map_user = {x: y for x, y in zip(unique_users, user_sequence)}
        self._reverse_user = {self._map_user[x]: x for x in self._map_user}

        unique_items = self.data.item.unique()
        unique_items.sort()
        item_sequence = range(self.data.item.nunique())
        self._map_item = {x: y for x, y in zip(unique_items, item_sequence)}
        self._reverse_item = {self._map_item[x]: x for x in self._map_item}
        self.data['user'] = self.data['user'].map(self._map_user)
        self.data['item'] = self.data['item'].map(self._map_item)
        self._reverse_this = self._reverse_user
        if not self.user_based:
            self._reverse_this = self._reverse_item

    def _mean_centering(self):
        if self._centering:
            r_u_means = self.data.groupby(self._centering).rating.mean()
            mean_ratings = r_u_means[self.data[self._centering]].values
            self.data['rating'] = (self.data.rating.values - mean_ratings)
            self.data = self.data[self.data.rating != 0]

    def x_y(self, x, y):
        if self.user_based:
            x, y = map(self._map_user.get, [x, y])
        else:
            x, y = map(self._map_item.get, [x, y])

        product_x_y = self.train[x].dot(self.train[y].T)
        if product_x_y[0, 0] == 0:
            return float(0)
        norm_x = sparse_norm(self.train[x])
        norm_y = sparse_norm(self.train[y])
        return (product_x_y/(norm_x*norm_y))[0, 0]

    def to_csv(self, csv_name=None):
        start = timer()
        rows, columns = 'user', 'item'
        if not self.user_based:
            rows, columns = columns, rows
        if not csv_name:
            csv_name = rows + "_cosine.csv"
        product = self.train.dot(self.train.T)
        norms = sparse_norm(self.train, axis=1)
        norms = sp.csr_matrix(norms.reshape((norms.size, 1)))
        chunk_size = 100
        init = product.shape[0] % chunk_size
        sim = np.true_divide(product[0:init], norms[0:init].dot(norms.T))
        sim[~np.isfinite(sim)] = 0
        sim = sp.csr_matrix(sim)
        previous = init
        for step in range(init + chunk_size, product.shape[0] + 1, chunk_size):
            cos = np.true_divide(product[previous:step],
                                 norms[previous:step].dot(norms.T))
            cos[~np.isfinite(cos)] = 0
            sim = sp.vstack((sim, sp.csr_matrix(cos)))
            previous = step
        nnz_upper = scipy.nonzero(sp.triu(A=sim, k=1, format='csr'))
        sim = np.array(sim[nnz_upper]).T.flatten()
        col1 = rows + "_1"
        col2 = rows + "_2"
        col3 = "similarity"
        df = pd.DataFrame({col1: nnz_upper[0], col2: nnz_upper[1], col3: sim})
        df[col1] = df[col1].map(self._reverse_this)
        df[col2] = df[col2].map(self._reverse_this)
        df[[col1, col2, col3]].to_csv(csv_name, index=False)
        timer(start)


class _CommonSearchSim(BaseMultiprocessing):
    _centering = None

    def __init__(self, data, user_based):
        BaseMultiprocessing.__init__(self)
        self.user_based = user_based
        group_by, other = 'user', 'item'
        if not self.user_based:
            group_by, other = other, group_by
        self.group_by = group_by
        self.other = other
        self.train = self._csv_to_dictionary(data)

    def _csv_to_dictionary(self, data):
        if self._centering:
            r_u_means = data.groupby(self._centering).rating.mean()
            mean_ratings = r_u_means[data[self._centering]].values
            data['rating'] = (data.rating.values - mean_ratings)

        unique_keys = data[self.group_by].unique()
        train = dict.fromkeys(unique_keys)

        for i in unique_keys:
            train[i] = {}

        iter_data = zip(data[self.group_by].values,
                        data[self.other].values,
                        data.rating.values)
        for u_m, m_u, r in iter_data:
            train[u_m][m_u] = r

        return train

    @staticmethod
    def x_y(x_ratings, y_ratings):
        common_u_i = set(x_ratings).intersection(y_ratings)
        if common_u_i:
            sum_xy = 0
            sum_sq_x = 0
            sum_sq_y = 0
            for common in common_u_i:
                sum_xy += x_ratings[common] * y_ratings[common]
                sum_sq_x += x_ratings[common] ** 2
                sum_sq_y += y_ratings[common] ** 2
            if sum_xy:
                return sum_xy / math.sqrt(sum_sq_x * sum_sq_y)

    def _partial_to_csv(self, x, list_y, csv_name):
        x_ratings = self.train[x]
        with open(csv_name, "a") as f:
            for y in list_y:
                y_ratings = self.train[y]
                sim = self.x_y(x_ratings, y_ratings)
                if sim:
                    csv.writer(f).writerow([x, y, sim])

    def to_csv(self, csv_name=None):
        start = timer()
        if not csv_name:
            csv_name = self.group_by + "_modified_cosine.csv"
        with open(csv_name, "w") as f:
            col1 = self.group_by + "_1"
            col2 = self.group_by + "_2"
            col3 = "similarity"
            csv.writer(f).writerow([col1, col2, col3])
        unique_u_i = sorted(self.train)
        unique_u_i_deque = deque(unique_u_i[:])
        for u_i_1 in unique_u_i:
            unique_u_i_deque.popleft()
            process_kwargs = {'x': u_i_1,
                              'list_y': unique_u_i_deque,
                              'csv_name': csv_name}
            self.new_process(self._partial_to_csv, **process_kwargs)
            if self.current_active_processes == self.max_active_processes:
                self.synchronize()
        if self.current_active_processes > 0:
            self.synchronize()
        timer(start)
