import time 
import collections
import random 

import numpy as np 
from numpy import ma
import scipy.sparse as sparse
import scipy.sparse.linalg as slinalg
import scipy.optimize as opt
from scipy import misc



def convert_to_sparse_mat(data, square = False, dim = 0):
    # data n x 3
    # return matrix m such that
    #   m[ data[i,0], data[i,1] ] = data[i,2] for all i = 1,...,n
    # as the data file is written by matlab
    # the index starts from 1, hence it is deducted by 1
    ndims = [int(np.max(data[:,0])), int(np.max(data[:,1]))]

    if square and ndims[0] != ndims[1]:
        if ndims[1-dim] > ndims[dim]:
            m = sparse.csr_matrix( (data[:,2], (data[:,0]-1, data[:,1]-1)), ndims )            
            ndims[1-dim] = ndims[dim]
            m = m[:ndims[0],:ndims[1]]
        else:
            ndims[1-dim] = ndims[dim]
            m = sparse.csr_matrix( (data[:,2], (data[:,0]-1, data[:,1]-1)), ndims )
    else:
        m = sparse.csr_matrix( (data[:,2], (data[:,0]-1, data[:,1]-1)), ndims )

    return m



def make_feature_smat_same_csr_format_as_incidence_smat(feat_smat, incidence_smat):
    data = np.zeros(incidence_smat.data.shape)

    for i in range(incidence_smat.shape[0]):
        ones_idxs = incidence_smat.indices[incidence_smat.indptr[i] : incidence_smat.indptr[i+1]]
        data[incidence_smat.indptr[i] : incidence_smat.indptr[i+1]] = feat_smat[i,ones_idxs].toarray().squeeze()

    ret_mat = sparse.csr_matrix( (data, incidence_smat.indices, incidence_smat.indptr), 
                        shape = incidence_smat.shape )
    return ret_mat


def append_zero_to_csr_matrix(csr_mat, n_zero_col=1, n_zero_row=1):
    indptr = np.concatenate([csr_mat.indptr, [csr_mat.indptr[-1]] * n_zero_row ])
    
    zero_appended_csr_mat = sparse.csr_matrix( (csr_mat.data, csr_mat.indices, indptr), 
                                            shape=(csr_mat.shape[0] + n_zero_row, 
                                                   csr_mat.shape[1] + n_zero_col) )
    return zero_appended_csr_mat

