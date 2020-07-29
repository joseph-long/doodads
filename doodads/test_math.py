import numpy as np
from .math import modified_gram_schmidt

def test_modified_gram_schmidt():
    init_x = np.asarray([
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 0]
    ])
    result = modified_gram_schmidt(init_x)

    ref = np.asarray([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    assert np.allclose(result, ref), "Last column should be zeroed by MGS"
