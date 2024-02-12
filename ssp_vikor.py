import itertools
import numpy as np

from mcda_method import MCDA_method


# VIKOR ascertaining Strong Sustainability Paradigm (SSP_VIKOR) method
class SSP_VIKOR(MCDA_method):
    def __init__(self, normalization_method = None, v = 0.5):
        """Create the VIKOR method object.

        Parameters
        -----------
            normalization_method : function
                VIKOR does not use normalization by default, thus `normalization_method` is set to None by default.
                However, you can choose method for normalization of decision matrix chosen `normalization_method` from `normalizations`.
                It is used in a way `normalization_method(X, types)` where `X` is a decision matrix
                and `types` is a vector with criteria types where 1 means profit and -1 means cost.
            v : float
                parameter that is weight of strategy of the majority of criteria (the maximum group utility)
        """
        self.v = v
        self.normalization_method = normalization_method


    def __call__(self, matrix, weights, types, s_coeff = 0):
        """
        Score alternatives provided in decision matrix `matrix` using criteria `weights` and criteria `types`.
        
        Parameters
        -----------
            matrix : ndarray
                Decision matrix including `m` alternatives in rows and `n` criteria in columns.
            weights: ndarray
                Vector including criteria weights. Sum of weights must be equal to 1.
            types: ndarray
                Vector including criteria types. Profit criteria are represented by 1 and cost by -1.
            s_coeff: ndarray
                Vector including values of sustainability coefficient for each criterion. It takes values
                from 0 to 1. 0 means full criteria compensation, and 1 represents a complete reduction
                of criteria compensation. It is recommended to set the `s_coeff` value in the range from 0 to 0.5.
        
        Returns
        --------
            ndrarray
                Vector with preference values of each alternative. The best alternative has the lowest preference value.
        
        Examples
        ---------
        >>> ssp_vikor = SSP_VIKOR(normalization_method = minmax_normalization)
        >>> pref = ssp_vikor(matrix, weights, types, s_coeff = s_set)
        >>> rank = rank_preferences(pref, reverse = False) 
        """

        SSP_VIKOR._verify_input_data(matrix, weights, types)
        return SSP_VIKOR._ssp_vikor(self, matrix, weights, types, self.normalization_method, self.v, s_coeff)


    # function for applying the SSP paradigm
    def _equalization(self, matrix, types, s_coeff):
        # Calculate the mean deviation values of the performance values in matrix.
        mad = (matrix - np.mean(matrix, axis = 0)) * s_coeff

        # Set as 0, those mean deviation values that for profit criteria are lower than 0
        # and those mean deviation values that for cost criteria are higher than 0
        for j, i in itertools.product(range(matrix.shape[1]), range(matrix.shape[0])):
            # for profit criteria
            if types[j] == 1:
                if mad[i, j] < 0:
                    mad[i, j] = 0
            # for cost criteria
            elif types[j] == -1:
                if mad[i, j] > 0:
                    mad[i, j] = 0

        # Subtract from performance values in decision matrix standard deviation values multiplied by a sustainability coefficient.
        return matrix - mad


    @staticmethod
    def _ssp_vikor(self, matrix, weights, types, normalization_method, v, s_coeff):

        # reducing compensation in decision matrix
        e_matrix = self._equalization(matrix, types, s_coeff)
        
        # Without special normalization method
        if normalization_method == None:

            # Determine the best `fstar` and the worst `fmin` values of all criterion function
            maximums_matrix = np.amax(e_matrix, axis = 0)
            minimums_matrix = np.amin(e_matrix, axis = 0)

            fstar = np.zeros(e_matrix.shape[1])
            fmin = np.zeros(e_matrix.shape[1])

            # for profit criteria (`types` == 1) and for cost criteria (`types` == -1)
            fstar[types == 1] = maximums_matrix[types == 1]
            fstar[types == -1] = minimums_matrix[types == -1]
            fmin[types == 1] = minimums_matrix[types == 1]
            fmin[types == -1] = maximums_matrix[types == -1]

            # Calculate the weighted matrix
            weighted_matrix = weights * ((fstar - e_matrix) / (fstar - fmin))

        else:
            # With chosen normalization method
            norm_matrix = normalization_method(matrix, types)

            fstar = np.amax(e_matrix, axis = 0)
            fmin = np.amin(e_matrix, axis = 0)

            # Calculate the weighted matrix
            weighted_matrix = weights * ((fstar - e_matrix) / (fstar - fmin))

        # Calculate the `S` and `R` values
        S = np.sum(weighted_matrix, axis = 1)
        R = np.amax(weighted_matrix, axis = 1)

        # Calculate the Q values
        Sstar = np.min(S)
        Smin = np.max(S)
        Rstar = np.min(R)
        Rmin = np.max(R)
        Q = v * (S - Sstar) / (Smin - Sstar) + (1 - v) * (R - Rstar) / (Rmin - Rstar)

        return Q