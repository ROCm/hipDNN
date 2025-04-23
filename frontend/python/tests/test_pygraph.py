# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import hipdnn

class TestPygraph:

    def test_basic_pygraph_creation(self):
        graph = hipdnn.pygraph(name="test_graph",
                               intermediate_data_type=hipdnn.data_type.FLOAT,
                               compute_data_type=hipdnn.data_type.FLOAT)
        assert graph.get_name() == "test_graph"
        assert graph.get_intermediate_data_type() == hipdnn.data_type.FLOAT
        assert graph.get_compute_data_type() == hipdnn.data_type.FLOAT
        assert graph.get_io_data_type() == hipdnn.data_type.NOT_SET

    def test_batchnorm_inference_node_creation(self):
        graph = hipdnn.pygraph(name="test_graph")

        # TODO
        #  - could use graph.tensor or graph.tensor_like() but not implemented yet
        #  - could also support pytorch tensors (dlpack)
        x = (hipdnn.tensor()
             .set_data_type(hipdnn.data_type.FLOAT)
             .set_dim([1, 2, 3, 4])
             .set_stride([5, 6, 7, 8]))
        mean = hipdnn.tensor()
        inv_variance = hipdnn.tensor()
        scale = hipdnn.tensor()
        bias = hipdnn.tensor()

        y = graph.batchnorm_inference(x, mean, inv_variance, scale, bias, name="BatchnormNode")
        assert y.get_name() == "BatchnormNode::Y"
        assert y.get_is_virtual() is True
        graph.validate() # throws exception if graph is not valid

    def test_pointwise_node_creation_single_input(self):
        graph = hipdnn.pygraph(name="test_graph")

        in_0 = (hipdnn.tensor()
                .set_data_type(hipdnn.data_type.FLOAT)
                .set_dim([1, 2, 3, 4])
                .set_stride([5, 6, 7, 8]))

        out_0 = graph.relu(in_0, name="PointwiseNode")
        assert out_0.get_name() == "PointwiseNode::OUT_0"
        assert out_0.get_is_virtual() is True
        graph.validate()

    def test_build_batchnorm_inference_graph(self):
        graph = hipdnn.pygraph(name="test_graph")

        x = (hipdnn.tensor()
             .set_uid(1)
             .set_name("X")
             .set_data_type(hipdnn.data_type.FLOAT)
             .set_dim([1, 2, 3, 4])
             .set_stride([5, 6, 7, 8]))
        mean = hipdnn.tensor().set_uid(2).set_name("Mean").set_data_type(hipdnn.data_type.FLOAT)
        inv_variance = hipdnn.tensor().set_uid(3).set_name("InvVariance").set_data_type(hipdnn.data_type.FLOAT)
        scale = hipdnn.tensor().set_uid(4).set_name("Scale").set_data_type(hipdnn.data_type.FLOAT)
        bias = hipdnn.tensor().set_uid(5).set_name("Bias").set_data_type(hipdnn.data_type.FLOAT)

        y = graph.batchnorm_inference(x, mean, inv_variance, scale, bias, name="BatchnormNode")
        graph.validate()
        graph.build_operation_graph()

    def test_build_batchnorm_backward_graph(self):
        graph = hipdnn.pygraph(name="BatchnormBackwardGraph",
                               io_data_type=hipdnn.data_type.FLOAT,
                               intermediate_data_type=hipdnn.data_type.HALF,
                               compute_data_type=hipdnn.data_type.FLOAT)

        dy = (hipdnn.tensor()
             .set_uid(1)
             .set_name("Dy")
             .set_data_type(hipdnn.data_type.FLOAT)
             .set_dim([1, 2, 3, 4])
             .set_stride([5, 6, 7, 8]))

        x = (hipdnn.tensor()
             .set_uid(2)
             .set_name("X")
             .set_data_type(hipdnn.data_type.FLOAT)
             .set_dim([1, 2, 3, 4])
             .set_stride([5, 6, 7, 8]))

        scale = hipdnn.tensor().set_uid(3).set_name("Scale").set_data_type(hipdnn.data_type.FLOAT)
        mean = hipdnn.tensor().set_uid(4).set_name("Mean").set_data_type(hipdnn.data_type.FLOAT)
        inv_variance = hipdnn.tensor().set_uid(5).set_name("InvVariance").set_data_type(hipdnn.data_type.FLOAT)

        dx, dscale, dbias = graph.batchnorm_backward(dy, x, scale, mean, inv_variance, name="BatchnormNode")

        graph.validate()
        graph.build_operation_graph()

    def test_build_multi_node_graph(self):
        graph = hipdnn.pygraph(name="MultiNodeGraphTest",
                               io_data_type=hipdnn.data_type.FLOAT,
                               intermediate_data_type=hipdnn.data_type.FLOAT,
                               compute_data_type=hipdnn.data_type.FLOAT)

        x = (hipdnn.tensor()
             .set_uid(1)
             .set_name("X")
             .set_data_type(hipdnn.data_type.FLOAT)
             .set_dim([1, 2, 3, 4])
             .set_stride([5, 6, 7, 8]))
        mean = hipdnn.tensor().set_uid(2).set_name("Mean").set_data_type(hipdnn.data_type.FLOAT)
        inv_variance = hipdnn.tensor().set_uid(3).set_name("InvVariance").set_data_type(hipdnn.data_type.FLOAT)
        scale = hipdnn.tensor().set_uid(4).set_name("Scale").set_data_type(hipdnn.data_type.FLOAT)
        bias = hipdnn.tensor().set_uid(5).set_name("Bias").set_data_type(hipdnn.data_type.FLOAT)

        y = graph.batchnorm_inference(x, mean, inv_variance, scale, bias, name="BatchnormNode")
        out_0 = graph.relu(y, name="PointwiseNode")
        graph.validate()
        graph.build_operation_graph()

    def test_single_node_serialization(self):
        graph = hipdnn.pygraph(name="SerializedGraph")

        x = (hipdnn.tensor()
             .set_data_type(hipdnn.data_type.FLOAT)
             .set_dim([1, 2, 3, 4])
             .set_stride([5, 6, 7, 8]))
        mean = hipdnn.tensor()
        inv_variance = hipdnn.tensor()
        scale = hipdnn.tensor()
        bias = hipdnn.tensor()

        y = graph.batchnorm_inference(x, mean, inv_variance, scale, bias, name="BatchnormNode")
        graph.validate()
        graph.build_operation_graph()
        data = graph.serialize()
        assert data is not None
        assert len(data) > 0

        deserialized_graph = hipdnn.pygraph(name="DeserializedGraph")
        deserialized_graph.deserialize(data)
        # TODO add more assertions to check the deserialized graph once deserialize() is implemented