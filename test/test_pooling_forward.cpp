#include "test_pooling_forward.hpp"

TEST(pooling_fwd, func_check_zero_padding) {
    test_2dpool_desc_t pool(1, 1, 
              224, 224,
              224 / 2, 224 / 2, 
               2, 2, 
               0, 0, 
               2, 2);
    Memory<float> srcData(pool.ih * pool.iw);
    Memory<float> dstDataCPU((224 / 2) * (224 / 2));
    Memory<float> dstDataGPU((224 / 2) * (224 / 2));
    populateMemoryRandom<float>(srcData);
    compute_hipdnn_maxpool_fwd<float>(pool, srcData.gpu(), dstDataGPU.gpu());
}
