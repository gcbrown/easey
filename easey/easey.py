from typing import Self

import numpy as np

try:
    from sparse_dot_mkl import csr_array, dot_product_mkl, gram_matrix_mkl

    using_mkl = True
except ImportError:
    from scipy.sparse import csr_array

    using_mkl = False


class EASE:
    def __init__(self, lambda_: float = 1) -> None:
        """Initialize EASE object with lambda as L2 regularization parameter."""
        self.lambda_ = lambda_

    def _get_G(self) -> np.ndarray:
        """Calculate the regularized gram matrix G = (X^T)X + λI."""
        if using_mkl:
            G = gram_matrix_mkl(self.X).toarray()
            # gram_matrix_mkl only returns the upper triangle
            G += np.triu(G, k=1).T
        else:
            G = (self.X.T @ self.X).toarray()

        # Add lambda to the diagonal
        G[np.diag_indices_from(G)] += self.lambda_
        return G

    def fit(self, df) -> Self:
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

    def predict(self, users: np.typing.ArrayLike, k: int = 10):
        """Predict the top-k most relevant items per user.

        Args:
            users: array-like of users to be scored
            k: predictions will be truncated to the top k

        Returns:
            Dataframe with 3 columns: user, item, score
        """
        # Get indexes of unique users - removes invalid or duplicate users
        user_idx = np.isin(self.U, users).nonzero()[0]
        if using_mkl:
            scores = dot_product_mkl(self.X[user_idx], self.B)
        else:
            scores = self.X[user_idx] @ self.B
        topk = np.argpartition(scores, -k)[:, -k:]

        return self.DataFrame({
            'user': self.U[np.repeat(user_idx, k)],
            'item': self.I[topk.flatten()],
            'score': np.take_along_axis(scores, topk, axis=-1).flatten(),
        })
