import numpy as np
import cython
import scipy.sparse

from cython cimport floating
from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

from libcpp.vector cimport vector
from libcpp.utility cimport pair

# https://groups.google.com/forum/#!topic/cython-users/H4UEM6IlvpM
cdef extern from "<algorithm>" namespace "std" nogil:
    void pop_heap[Iter, Compare](Iter first, Iter last, Compare comp)
    void push_heap[Iter, Compare](Iter first, Iter last, Compare comp)

cdef extern from "<functional>" namespace "std" nogil:
    cdef cppclass greater[T]:
        greater()


@cython.boundscheck(False)
def all_pairs_knn(items, int K=100):
    """ Returns the top K nearest neighbours for each row in the matrix """
    users = items.T.tocsr()

    cdef int[:] item_indptr = items.indptr, item_indices = items.indices
    cdef double[:] item_data = items.data

    cdef int[:] user_indptr = users.indptr, user_indices = users.indices
    cdef double[:] user_data = users.data

    cdef int item_count = items.shape[0]
    cdef int i, u, index1, index2, j, head, length
    cdef double w1, w2

    # TODO: malloc/free, instead of vector so can paralleralize (also we have fixed size)
    # TODO: unittest?

    cdef vector[double] sums
    cdef vector[int] nonzeros
    cdef vector[pair[double, int]] results

    nonzeros.resize(item_count, -1)
    sums.resize(item_count, 0)

    cdef greater[pair[double, int]] heap_order
    cdef pair[double, int] result

    values = np.zeros(item_count * K)
    rows = np.zeros(item_count * K, dtype=int)
    cols = np.zeros(item_count * K, dtype=int)

    cdef double[:] _values = values
    cdef long[:] _rows = rows
    cdef long[:] _cols = cols
    cdef int current = 0

    for i in range(item_count):
        head = -2
        length = 0

        for index1 in range(item_indptr[i], item_indptr[i+1]):
            u = item_indices[index1]
            w1 = item_data[index1]

            for index2 in range(user_indptr[u], user_indptr[u+1]):
                j = user_indices[index2]
                w2 = user_data[index2]
                sums[j] += w1 * w2

                if nonzeros[j] == -1:
                    nonzeros[j] = head
                    head = j
                    length += 1

        results.clear()

        for index1 in range(length):
            w1 = sums[head]
            index2 = head

            head = nonzeros[head]

            sums[index2] = 0
            nonzeros[index2] = -1

            if ((results.size() >= K) and
                    (w1 < results[0].first)):
                continue

            if results.size() >= K:
                pop_heap(results.begin(), results.end(), heap_order)
                results.pop_back()

            result.first = w1
            result.second = index2
            results.push_back(result)
            push_heap(results.begin(), results.end(), heap_order)

        # TODO: gil?
        for it in results:
            _rows[current] = i
            _cols[current] = it.second
            _values[current] = it.first
            current += 1

    _rows = _cols = _values = None
    rows.resize(current)
    cols.resize(current)
    values.resize(current)

    return scipy.sparse.coo_matrix((values, (rows, cols)), shape=(item_count, item_count))
