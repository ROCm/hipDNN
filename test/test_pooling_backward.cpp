#include "test_pooling_common.hpp"

TEST(pooling_backward, func_check_pooling_stride_2x2){
    int oheight = 4 / 2 + 1, owidth = 4 / 2 + 1;
    test_pooling_t test_case(1, 1, 4, 4,
                             2, 2, 
                             0, 0, 
                             2, 2,
                             1, 1,
                             oheight, owidth);
    Memory<float> dataSrc(test_case.in * test_case.ichannel * test_case.iheight * test_case.iwidth);
    Memory<float> dataGrad(test_case.in * test_case.ichannel * test_case.iheight * test_case.iwidth);
    populateMemoryRandom(dataSrc);
    populateMemoryRandom(dataGrad);
    Memory<float> dataDst;
    compute_hipdnn_conv_fwd(test_case, dataSrc.gpu(), dataGrad.gpu(),dataDst.gpu());
}