import numpy as np

from daaf import npsci


def test_item():
    int_value = 1
    int_array = np.array(int_value)
    float_value = 1.0
    float_array = np.array(float_value)
    string_value = "z"
    string_array = np.array(string_value)
    assert npsci.item(int_value) == 1
    assert isinstance(npsci.item(int_array), int)
    assert npsci.item(int_array) == 1
    assert npsci.item(float_value) == 1.0
    assert isinstance(npsci.item(float_array), float)
    assert npsci.item(float_array) == 1.0
    assert npsci.item(string_value) == "z"
    assert isinstance(npsci.item(string_array), str)
    assert npsci.item(string_array) == "z"
