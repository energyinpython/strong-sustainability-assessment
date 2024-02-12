import itertools
import numpy as np
from mcda_method import MCDA_method

class SPOTIS_SSP(MCDA_method):
    def __init__(self):
        """Create SPOTIS_SSP method object.
        """
        pass


    def __call__(self, matrix, weights, types, s_coeff = 0):
        """Score alternatives provided in decision matrix `matrix` using criteria `weights` and criteria `types`.

        Parameters
        -----------
            matrix : ndarray
                Decision matrix with m alternatives in rows and n criteria in columns.
            weights: ndarray
                Vector with criteria weights. Sum of weights must be equal to 1.
            types: ndarray
                Vector with criteria types. Profit criteria are represented by 1 and cost by -1.
            s_coeff: ndarray
                Vector including values of sustainability coefficient for each criterion. It takes values
                from 0 to 1. 0 means full criteria compensation, and 1 represents a complete reduction
                of criteria compensation. It is recommended to set the `s_coeff` value in the range from 0 to 0.5.

        Returns
        --------
            ndrarray
                Vector with preference values of each alternative. The best alternative has the lowest preference value. 

        Examples
        ----------
        >>> spotis_ssp = SPOTIS_SSP()
        >>> pref = spotis_ssp(matrix, weights, types)
        >>> rank = rank_preferences(pref, reverse = False)
        """
        SPOTIS_SSP._verify_input_data(matrix, weights, types)
        return SPOTIS_SSP._spotis_ssp(self, matrix, weights, types, s_coeff)
    
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
    def _spotis_ssp(self, matrix, weights, types, s_coeff):
        # Create the matrix with reduced compensation
        e_matrix = self._equalization(matrix, types, s_coeff)

        # Calculate maximum and minimum bounds with reduced compensation based on matrix with reduced compensation
        bounds_min = np.amin(e_matrix, axis = 0)
        bounds_max = np.amax(e_matrix, axis = 0)
        bounds = np.vstack((bounds_min, bounds_max))

        # Determine Ideal Solution Point (ISP)
        isp = np.zeros(e_matrix.shape[1])
        isp[types == 1] = bounds[1, types == 1]
        isp[types == -1] = bounds[0, types == -1]

        # Calculate normalized distances
        norm_matrix = np.abs(matrix - isp) / np.abs(bounds[1, :] - bounds[0, :])
        # Calculate the normalized weighted average distance
        D = np.sum(weights * norm_matrix, axis = 1)
        return D