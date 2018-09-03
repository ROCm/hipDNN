#include "test_pooling_fwd_common.hpp"
/*
    struct test_2dpool_desc_t {
    int mb, c; // Minibatch and channels
    int ih, iw; // input dimensions
    int oh, ow; // output dimensions
    int kh, kw; // kernel dimensions
    int padt, padl; // padding dimensions
    int strh, strw; // stride dimensions

    test_2dpool_desc_t(int mb, int c, 
                       int ih, int iw, 
                       int oh, int ow,
                       int kh, int kw,
                       int padt, int padl,
                       int strh, int strw
                    ): mb(mb), c(c),
                       ih(ih), iw(iw),
                       oh(oh), ow(ow),
                       kh(kh), kw(kw),
                       padt(padt), padl(padl),
                       strh(strh), strw(strw){}

}; */

TEST(pooling_fwd, func_check_zero_padding) {
    test_2dpool_desc_t pool(1, 3, 
              224, 224,
              224 / 2, 224 / 2, 
               2, 2, 
               0, 0, 
               2, 2);

    Memory<float> srcData(pool.ih * pool.iw);
    Memory<float> dstDataCPU((224 / 2) * (224 / 2));
    Memory<float> dstDataGPU((224 / 2) * (224 / 2));
    populateMemoryRandom<float>(srcData);
    compute_cpuref_maxpool_fwd<float>(pool, srcData.cpu(), dstDataCPU.cpu());
    compute_hipdnn_maxpool_fwd<float>(pool, srcData.gpu(), dstDataGPU.gpu());
    Equals<float>(dstDataCPU, dstDataGPU);
    EXPECT_EQ(0, 0);
}
