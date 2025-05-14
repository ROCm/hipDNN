# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import hipdnn
import numpy as np

class TestTensorAttributes:

    def test_tensor_attribute_accessors(self):
        tensor = hipdnn.tensor()
        tensor.set_uid(12345)
        tensor.set_name("test_tensor")
        tensor.set_data_type(hipdnn.data_type.FLOAT)
        tensor.set_stride([1, 2, 3])
        tensor.set_dim([4, 5, 6])
        tensor.set_is_virtual(False)

        assert tensor.get_uid() == 12345
        assert tensor.get_name() == "test_tensor"
        assert tensor.get_data_type() == hipdnn.data_type.FLOAT
        assert tensor.get_stride() == [1, 2, 3]
        assert tensor.get_dim() == [4, 5, 6]
        assert tensor.get_is_virtual() is False

    def test_tensor_attribute_chained_setters(self):
        tensor = (hipdnn.tensor()
                  .set_uid(12345)
                  .set_name("test_tensor")
                  .set_data_type(hipdnn.data_type.FLOAT)
                  .set_stride([1, 2, 3])
                  .set_dim([4, 5, 6])
                  .set_is_virtual(False))

        assert tensor.get_uid() == 12345
        assert tensor.get_name() == "test_tensor"
        assert tensor.get_data_type() == hipdnn.data_type.FLOAT
        assert tensor.get_stride() == [1, 2, 3]
        assert tensor.get_dim() == [4, 5, 6]
        assert tensor.get_is_virtual() is False

    def test_tensor_attribute_set_output(self):
        tensor = hipdnn.tensor()
        tensor.set_output(True)

        assert tensor.get_is_virtual() is False

    def test_tensor_attribute_clear_uid(self):
        tensor = hipdnn.tensor()
        tensor.set_uid(12345)
        tensor.clear_uid()

        assert tensor.get_uid() == 0

    def test_tensor_attribute_numpy(self):
        strides = np.array([1, 2, 3])
        dimensions = np.array([4, 5, 6])
        tensor = hipdnn.tensor().set_stride(strides).set_dim(dimensions)

        assert np.array_equal(tensor.get_stride(), strides)
        assert np.array_equal(tensor.get_dim(), dimensions)
