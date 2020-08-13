r"""This module provides a fast implementation of the Instrumentd PCA
algorithm [1]_ [2]_.

In a nutshell, IPCA seeks to minimize the following objective function

.. math::
    :nowrap:

    \begin{eqnarray}
        \min_{\Gamma,\{F_t\}} & \sum_t
        \left( Y_t - \hat Y_t \right)^T \cdot \left( Y_t - \hat Y_t \right) \\

        \text{s.t.} & \Gamma^T \cdot \Gamma = I \\

        \text{where} & \hat{Y}_t = X_t \cdot \Gamma \cdot F_t
    \end{eqnarray}

References
----------
.. [1] Bryan T. Kelly, Seth Pruitt, Yinan Su,
   "Characteristics are covariances: A unified model of risk and return",
   Journal of Financial Economics, vol. 134, issue 3, pp. 501-524, 2019.
.. [2] Bryan T. Kelly, Seth Pruitt, Yinan Su,
   "Instrumented Principal Component Analysis",
   https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2983919
"""
from .fastipca import train, predict

__all__ = ["train", "predict"]
