import numpy as np
from sparse_dot_mkl import csr_array, gram_matrix_mkl


class EASE:
    def __init__(self, lambda_: float = 1):
        """Initialize EASE object with lambda as L2 regularization parameter."""
        self.lambda_ = lambda_

    def _get_G(self):
        """Calculate the regularized gram matrix G = (X^T)X + λI."""
        G = gram_matrix_mkl(self.X, dense=True)
        # gram_matrix_mkl only returns the upper triangle
        G += np.triu(G, k=1).T

        # Add lambda to the diagonal
        G[np.diag_indices_from(G)] += self.lambda_
        return G

    def fit(self, df):
        """Fit the EASE model to user x item interactions.

        Args:
            df: A dataframe with 3 columns: user, item, rating.

        Returns:
            self
        """
        # Record the dataframe type so we can use it when returning predictions
        self.DataFrame = type(df)
        # Get unique users/items and indexes for use in the sparse interaction matrix
        self.U, user_idx = np.unique(df['user'], return_inverse=True)
        self.I, item_idx = np.unique(df['item'], return_inverse=True)

        # From Algorithm 1 in the EASE paper
        self.X = csr_array((df['rating'], (user_idx, item_idx)))
        G = self._get_G()
        P = np.linalg.inv(G)
        self.B = P / (-np.diag(P))
        np.fill_diagonal(self.B, 0)
        return self
