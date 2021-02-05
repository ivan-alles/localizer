# Copyright 2018-2020 Ivan Alles. See also the LICENSE file.

import numpy as np

from localizer import utils


def test_make_transform2():
    assert np.allclose(utils.make_transform2(2, 0, 10, 20), [[2, 0, 10], [0, 2, 20], [0, 0, 1]])
    a = 0.2
    sa = np.sin(a)
    ca = np.cos(a)
    assert np.allclose(utils.make_transform2(1, a, 10, 20), [[ca, -sa, 10], [sa, ca, 20], [0, 0, 1]])
