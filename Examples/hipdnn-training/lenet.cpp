/*
 * This code is released into the public domain.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include <hipblas.h>
#include <hipdnn.h>

#include "readubyte.h"

///////////////////////////////////////////////////////////////////////////////////////////
// Definitions and helper utilities

// Block width for CUDA kernels
#define BW 128

#ifdef USE_GFLAGS
    #include <gflags/gflags.h>

    #ifndef _WIN32
        #define gflags google
    #endif
#else
    // Constant versions of gflags
    #define DEFINE_int32(flag, default_value, description) const int FLAGS_##flag = (default_value)
    #define DEFINE_uint64(flag, default_value, description) const unsigned long long FLAGS_##flag = (default_value)
    #define DEFINE_bool(flag, default_value, description) const bool FLAGS_##flag = (default_value)
    #define DEFINE_double(flag, default_value, description) const double FLAGS_##flag = (default_value)
    #define DEFINE_string(flag, default_value, description) const std::string FLAGS_##flag ((default_value))
#endif

/**
 * Computes ceil(x / y) for integral nonnegative values.
 */
static inline unsigned int RoundUp(unsigned int nominator, unsigned int denominator)
{
    return (nominator + denominator - 1) / denominator;
}

/**
 * Saves a PGM grayscale image out of unsigned 8-bit data
 */
void SavePGMFile(const unsigned char *data, size_t width, size_t height, const char *filename)
{
    FILE *fp = fopen(filename, "wb");
    if (fp)
    {
        fprintf(fp, "P5\n%lu %lu\n255\n", width, height);
        fwrite(data, sizeof(unsigned char), width * height, fp);
        fclose(fp);
    }
}

//////////////////////////////////////////////////////////////////////////////
// Error handling

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    hipDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkHIPDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != HIPDNN_STATUS_SUCCESS) {                              \
      _error << "HIPDNN failure: " << hipdnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkHIPErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)


///////////////////////////////////////////////////////////////////////////////////////////
// Command-line flags

// Application parameters
DEFINE_int32(gpu, 0, "The GPU ID to use");
DEFINE_int32(iterations, 1000, "Number of iterations for training");
DEFINE_int32(random_seed, -1, "Override random seed (default uses std::random_device)");
DEFINE_int32(classify, -1, "Number of images to classify to compute error rate (default uses entire test set)");

// Batch parameters
DEFINE_uint64(batch_size, 32, "Batch size for training");

// Filenames
DEFINE_bool(pretrained, false, "Use the pretrained HIPDNN model as input");
DEFINE_bool(save_data, false, "Save pretrained weights to file");
DEFINE_string(train_images, "train-images-idx3-ubyte", "Training images filename");
DEFINE_string(train_labels, "train-labels-idx1-ubyte", "Training labels filename");
DEFINE_string(test_images, "t10k-images-idx3-ubyte", "Test images filename");
DEFINE_string(test_labels, "t10k-labels-idx1-ubyte", "Test labels filename");

// Solver parameters
DEFINE_double(learning_rate, 0.01, "Base learning rate");
DEFINE_double(lr_gamma, 0.0001, "Learning rate policy gamma");
DEFINE_double(lr_power, 0.75, "Learning rate policy power");


///////////////////////////////////////////////////////////////////////////////////////////
// Layer representations

/**
 * Represents a convolutional layer with bias.
 */
struct ConvBiasLayer
{
    int in_channels, out_channels, kernel_size;
    int in_width, in_height, out_width, out_height;

    std::vector<float> pconv, pbias;

    ConvBiasLayer(int in_channels_, int out_channels_, int kernel_size_,
                  int in_w_, int in_h_) : pconv(in_channels_ * kernel_size_ * kernel_size_ * out_channels_),
                  pbias(out_channels_)
    {
        in_channels = in_channels_;
        out_channels = out_channels_;
        kernel_size = kernel_size_;
        in_width = in_w_;
        in_height = in_h_;
        out_width = in_w_ - kernel_size_ + 1;
        out_height = in_h_ - kernel_size_ + 1;
    }

    bool FromFile(const char *fileprefix)
    {
        std::stringstream ssf, ssbf;
        ssf << fileprefix << ".bin";
        ssbf << fileprefix << ".bias.bin";

        // Read weights file
        FILE *fp = fopen(ssf.str().c_str(), "rb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
            return false;
        }
        fread(&pconv[0], sizeof(float), in_channels * out_channels * kernel_size * kernel_size, fp);
        fclose(fp);

        // Read bias file
        fp = fopen(ssbf.str().c_str(), "rb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
            return false;
        }
        fread(&pbias[0], sizeof(float), out_channels, fp);
        fclose(fp);
        return true;
    }

    void ToFile(const char *fileprefix)
    {
        std::stringstream ssf, ssbf;
        ssf << fileprefix << ".bin";
        ssbf << fileprefix << ".bias.bin";

        // Write weights file
        FILE *fp = fopen(ssf.str().c_str(), "wb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
            exit(2);
        }
        fwrite(&pconv[0], sizeof(float), in_channels * out_channels * kernel_size * kernel_size, fp);
        fclose(fp);

        // Write bias file
        fp = fopen(ssbf.str().c_str(), "wb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
            exit(2);
        }
        fwrite(&pbias[0], sizeof(float), out_channels, fp);
        fclose(fp);
    }
};

/**
 * Represents a max-pooling layer.
 */
struct MaxPoolLayer
{
    int size, stride;
    MaxPoolLayer(int size_, int stride_) : size(size_), stride(stride_) {}
};

/**
 * Represents a fully-connected neural network layer with bias.
 */
struct FullyConnectedLayer
{
    int inputs, outputs;
    std::vector<float> pneurons, pbias;

    FullyConnectedLayer(int inputs_, int outputs_) : outputs(outputs_), inputs(inputs_),
        pneurons(inputs_ * outputs_), pbias(outputs_) {}

    bool FromFile(const char *fileprefix)
    {
        std::stringstream ssf, ssbf;
        ssf << fileprefix << ".bin";
        ssbf << fileprefix << ".bias.bin";

        // Read weights file
        FILE *fp = fopen(ssf.str().c_str(), "rb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
            return false;
        }
        fread(&pneurons[0], sizeof(float), inputs * outputs, fp);
        fclose(fp);

        // Read bias file
        fp = fopen(ssbf.str().c_str(), "rb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
            return false;
        }
        fread(&pbias[0], sizeof(float), outputs, fp);
        fclose(fp);
        return true;
    }

    void ToFile(const char *fileprefix)
    {
        std::stringstream ssf, ssbf;
        ssf << fileprefix << ".bin";
        ssbf << fileprefix << ".bias.bin";

        // Write weights file
        FILE *fp = fopen(ssf.str().c_str(), "wb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
            exit(2);
        }
        fwrite(&pneurons[0], sizeof(float), inputs * outputs, fp);
        fclose(fp);

        // Write bias file
        fp = fopen(ssbf.str().c_str(), "wb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
            exit(2);
        }
        fwrite(&pbias[0], sizeof(float), outputs, fp);
        fclose(fp);
    }
};

///////////////////////////////////////////////////////////////////////////////////////////
// GPU Kernels

/**
 * Fills a floating-point array with ones.
 *
 * @param vec The array to fill.
 * @param size The number of elements in the array.
 */
__global__ void FillOnes(float *vec, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    vec[idx] = 1.0f;
}

/**
 * Computes the backpropagation results of the Softmax loss for each result in a batch.
 * Uses the softmax values obtained from forward propagation to compute the difference.
 *
 * @param label The training batch label values.
 * @param num_labels The number of possible labels.
 * @param batch_size The size of the trained batch.
 * @param diff The resulting gradient.
 */
__global__ void SoftmaxLossBackprop(const float *label, int num_labels, int batch_size, float *diff)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size)
        return;

    const int label_value = static_cast<int>(label[idx]);

    // For each item in the batch, decrease the result of the label's value by 1
    diff[idx * num_labels + label_value] -= 1.0f;
}

///////////////////////////////////////////////////////////////////////////////////////////
// HIPDNN/HIPBLAS training context

struct TrainingContext
{
    hipdnnHandle_t hipdnnHandle;
    hipblasHandle_t hipblasHandle;

    hipdnnTensorDescriptor_t dataTensor, conv1Tensor, conv1BiasTensor, pool1Tensor,
                             conv2Tensor, conv2BiasTensor, pool2Tensor, fc1Tensor, fc2Tensor;
    hipdnnFilterDescriptor_t conv1filterDesc, conv2filterDesc;
    hipdnnConvolutionDescriptor_t conv1Desc, conv2Desc;
    hipdnnConvolutionFwdAlgo_t conv1algo, conv2algo;
    hipdnnConvolutionBwdFilterAlgo_t conv1bwfalgo, conv2bwfalgo;
    hipdnnConvolutionBwdDataAlgo_t conv2bwdalgo;
    hipdnnPoolingDescriptor_t poolDesc;
    hipdnnActivationDescriptor_t fc1Activation;

    int m_gpuid;
    int m_batchSize;
    size_t m_workspaceSize;

    FullyConnectedLayer& ref_fc1, &ref_fc2;

    // Disable copying
    TrainingContext& operator=(const TrainingContext&) = delete;
    TrainingContext(const TrainingContext&) = delete;

    TrainingContext(int gpuid, int batch_size,
                    ConvBiasLayer& conv1, MaxPoolLayer& pool1, ConvBiasLayer& conv2, MaxPoolLayer& pool2,
                    FullyConnectedLayer& fc1, FullyConnectedLayer& fc2) : ref_fc1(fc1), ref_fc2(fc2), m_gpuid(gpuid)
    {
        m_batchSize = batch_size;

        // Create HIPBLAS and HIPDNN handles
        checkHIPErrors(hipSetDevice(gpuid));
        checkHIPErrors(hipblasCreate(&hipblasHandle));
        checkHIPDNN(hipdnnCreate(&hipdnnHandle));

        // Create tensor descriptors
        checkHIPDNN(hipdnnCreateTensorDescriptor(&dataTensor));
        checkHIPDNN(hipdnnCreateTensorDescriptor(&conv1Tensor));
        checkHIPDNN(hipdnnCreateTensorDescriptor(&conv1BiasTensor));
        checkHIPDNN(hipdnnCreateTensorDescriptor(&pool1Tensor));
        checkHIPDNN(hipdnnCreateTensorDescriptor(&conv2Tensor));
        checkHIPDNN(hipdnnCreateTensorDescriptor(&conv2BiasTensor));
        checkHIPDNN(hipdnnCreateTensorDescriptor(&pool2Tensor));
        checkHIPDNN(hipdnnCreateTensorDescriptor(&fc1Tensor));
        checkHIPDNN(hipdnnCreateTensorDescriptor(&fc2Tensor));

        checkHIPDNN(hipdnnCreateActivationDescriptor(&fc1Activation));

        checkHIPDNN(hipdnnCreateFilterDescriptor(&conv1filterDesc));
        checkHIPDNN(hipdnnCreateFilterDescriptor(&conv2filterDesc));

        checkHIPDNN(hipdnnCreateConvolutionDescriptor(&conv1Desc));
        checkHIPDNN(hipdnnCreateConvolutionDescriptor(&conv2Desc));

        checkHIPDNN(hipdnnCreatePoolingDescriptor(&poolDesc));

        // Set tensor descriptor sizes
        checkHIPDNN(hipdnnSetTensor4dDescriptor(conv1BiasTensor,
                                              HIPDNN_TENSOR_NCHW,
                                              HIPDNN_DATA_FLOAT,
                                              1, conv1.out_channels,
                                              1, 1));
        checkHIPDNN(hipdnnSetTensor4dDescriptor(conv2BiasTensor,
                                              HIPDNN_TENSOR_NCHW,
                                              HIPDNN_DATA_FLOAT,
                                              1, conv2.out_channels,
                                              1, 1));

        checkHIPDNN(hipdnnSetPooling2dDescriptor(poolDesc,
                                               HIPDNN_POOLING_MAX,
                                               HIPDNN_PROPAGATE_NAN,
                                               pool1.size, pool1.size,
                                               0, 0,
                                               pool1.stride, pool1.stride));
        checkHIPDNN(hipdnnSetTensor4dDescriptor(pool2Tensor,
                                              HIPDNN_TENSOR_NCHW,
                                              HIPDNN_DATA_FLOAT,
                                              batch_size, conv2.out_channels,
                                              conv2.out_height / pool2.stride,
                                              conv2.out_width / pool2.stride));

        checkHIPDNN(hipdnnSetTensor4dDescriptor(fc1Tensor,
                                              HIPDNN_TENSOR_NCHW,
                                              HIPDNN_DATA_FLOAT,
                                              batch_size, fc1.outputs, 1, 1));

        checkHIPDNN(hipdnnSetTensor4dDescriptor(fc2Tensor,
                                              HIPDNN_TENSOR_NCHW,
                                              HIPDNN_DATA_FLOAT,
                                              batch_size, fc2.outputs, 1, 1));

        checkHIPDNN(hipdnnSetActivationDescriptor(fc1Activation, HIPDNN_ACTIVATION_RELU,
                                                HIPDNN_PROPAGATE_NAN, 0.0, 0.0, 0.0));



        // Set convolution tensor sizes and compute workspace size
        size_t workspace = 0;
        workspace = std::max(workspace, SetFwdConvolutionTensors(conv1, dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, conv1algo));
        workspace = std::max(workspace, SetBwdConvolutionTensors(dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, &conv1bwfalgo, nullptr));

        workspace = std::max(workspace, SetFwdConvolutionTensors(conv2, pool1Tensor, conv2Tensor, conv2filterDesc, conv2Desc, conv2algo));
        workspace = std::max(workspace, SetBwdConvolutionTensors(pool1Tensor, conv2Tensor, conv2filterDesc, conv2Desc, &conv2bwfalgo, &conv2bwdalgo));

        // The workspace is allocated later (if necessary)
        m_workspaceSize = workspace;
    }

    ~TrainingContext()
    {
        checkHIPErrors(hipSetDevice(m_gpuid));

        checkHIPErrors(hipblasDestroy(hipblasHandle));
        checkHIPDNN(hipdnnDestroy(hipdnnHandle));
        checkHIPDNN(hipdnnDestroyTensorDescriptor(dataTensor));
        checkHIPDNN(hipdnnDestroyTensorDescriptor(conv1Tensor));
        checkHIPDNN(hipdnnDestroyTensorDescriptor(conv1BiasTensor));
        checkHIPDNN(hipdnnDestroyTensorDescriptor(pool1Tensor));
        checkHIPDNN(hipdnnDestroyTensorDescriptor(conv2Tensor));
        checkHIPDNN(hipdnnDestroyTensorDescriptor(conv2BiasTensor));
        checkHIPDNN(hipdnnDestroyTensorDescriptor(pool2Tensor));
        checkHIPDNN(hipdnnDestroyTensorDescriptor(fc1Tensor));
        checkHIPDNN(hipdnnDestroyTensorDescriptor(fc2Tensor));
        checkHIPDNN(hipdnnDestroyActivationDescriptor(fc1Activation));
        checkHIPDNN(hipdnnDestroyFilterDescriptor(conv1filterDesc));
        checkHIPDNN(hipdnnDestroyFilterDescriptor(conv2filterDesc));
        checkHIPDNN(hipdnnDestroyConvolutionDescriptor(conv1Desc));
        checkHIPDNN(hipdnnDestroyConvolutionDescriptor(conv2Desc));
        checkHIPDNN(hipdnnDestroyPoolingDescriptor(poolDesc));
    }

    size_t SetFwdConvolutionTensors(ConvBiasLayer& conv, hipdnnTensorDescriptor_t& srcTensorDesc, hipdnnTensorDescriptor_t& dstTensorDesc,
                                    hipdnnFilterDescriptor_t& filterDesc, hipdnnConvolutionDescriptor_t& convDesc,
                                    hipdnnConvolutionFwdAlgo_t& algo)
    {
        size_t sizeInBytes = 0;

        int n = m_batchSize;
        int c = conv.in_channels;
        int h = conv.in_height;
        int w = conv.in_width;
        int oc = conv.out_channels;
        int k  = conv.kernel_size;
        int nb = 4;
        const int filterDimA[4] = { conv.out_channels, conv.in_channels, conv.kernel_size, conv.kernel_size};

        checkHIPDNN(hipdnnSetTensor4dDescriptor(srcTensorDesc,
                                              HIPDNN_TENSOR_NCHW,
                                              HIPDNN_DATA_FLOAT,
                                              n, c,
                                              h, w));

        checkHIPDNN(hipdnnSetFilterNdDescriptor(filterDesc,
                                              HIPDNN_DATA_FLOAT,
                                              HIPDNN_TENSOR_NCHW,
                                              nb,
                                              filterDimA));;

        checkHIPDNN(hipdnnSetConvolution2dDescriptor(convDesc,
                                                   0, 0,
                                                   1, 1,
                                                   1, 1,
                                                   HIPDNN_CROSS_CORRELATION,
                                                   HIPDNN_DATA_FLOAT));


        // Find dimension of convolution output
        checkHIPDNN(hipdnnGetConvolution2dForwardOutputDim(convDesc,
                                                         srcTensorDesc,
                                                         filterDesc,
                                                         &n, &c, &h, &w));

        checkHIPDNN(hipdnnSetTensor4dDescriptor(dstTensorDesc,
                                              HIPDNN_TENSOR_NCHW,
                                              HIPDNN_DATA_FLOAT,
                                              n, c,
                                              h, w));

        checkHIPDNN(hipdnnGetConvolutionForwardAlgorithm(hipdnnHandle,
                                                       srcTensorDesc,
                                                       filterDesc,
                                                       convDesc,
                                                       dstTensorDesc,
                                                       HIPDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                       0,
                                                       &algo));

        checkHIPDNN(hipdnnGetConvolutionForwardWorkspaceSize(hipdnnHandle,
                                                           srcTensorDesc,
                                                           filterDesc,
                                                           convDesc,
                                                           dstTensorDesc,
                                                           algo,
                                                           &sizeInBytes));

        return sizeInBytes;
    }

    void ForwardPropagation(float *data, float *conv1, float *pool1, float *conv2, float *pool2, float *fc1, float *fc1relu,
                            float *fc2, float *result,
                            float *pconv1, float *pconv1bias,
                            float *pconv2, float *pconv2bias,
                            float *pfc1, float *pfc1bias,
                            float *pfc2, float *pfc2bias, void *workspace, float *onevec)
    {
        float alpha = 1.0f, beta = 0.0f;
        checkHIPErrors(hipSetDevice(m_gpuid));

        // Conv1 layer

        checkHIPDNN(hipdnnConvolutionForward(hipdnnHandle, &alpha, dataTensor,
                                           data, conv1filterDesc, pconv1, conv1Desc,
                                           conv1algo, workspace, m_workspaceSize, &beta,
                                           conv1Tensor, conv1));
										   
       checkHIPDNN(hipdnnAddTensor(hipdnnHandle, &alpha, conv1BiasTensor,
                                  pconv1bias, &alpha, conv1Tensor, conv1));


        // Pool1 layer
        checkHIPDNN(hipdnnPoolingForward(hipdnnHandle, poolDesc, &alpha, conv1Tensor,
                                       conv1, &beta, pool1Tensor, pool1, true));

        // Conv2 layer
        checkHIPDNN(hipdnnConvolutionForward(hipdnnHandle, &alpha, pool1Tensor,
                                           pool1, conv2filterDesc, pconv2, conv2Desc,
                                           conv2algo, workspace, m_workspaceSize, &beta,
                                           conv2Tensor, conv2));
        checkHIPDNN(hipdnnAddTensor(hipdnnHandle, &alpha, conv2BiasTensor,
                                  pconv2bias, &alpha, conv2Tensor, conv2));

        // Pool2 layer
        checkHIPDNN(hipdnnPoolingForward(hipdnnHandle, poolDesc, &alpha, conv2Tensor,
                                       conv2, &beta, pool2Tensor, pool2, true));

        // FC1 layer
        // Forward propagate neurons using weights (fc1 = pfc1'*pool2)
        checkHIPErrors(hipblasSgemm(hipblasHandle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                                    ref_fc1.outputs, m_batchSize, ref_fc1.inputs,
                                    &alpha,
                                    pfc1, ref_fc1.inputs,
                                    pool2, ref_fc1.inputs,
                                    &beta,
                                    fc1, ref_fc1.outputs));
        // Add bias using GEMM's "beta" (fc1 += pfc1bias*1_vec')
        checkHIPErrors(hipblasSgemm(hipblasHandle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                    ref_fc1.outputs, m_batchSize, 1,
                                    &alpha,
                                    pfc1bias, ref_fc1.outputs,
                                    onevec, 1,
                                    &alpha,
                                    fc1, ref_fc1.outputs));

        // ReLU activation
        checkHIPDNN(hipdnnActivationForward(hipdnnHandle, fc1Activation, &alpha,
                                          fc1Tensor, fc1, &beta, fc1Tensor, fc1relu));

        // FC2 layer
        // Forward propagate neurons using weights (fc2 = pfc2'*fc1relu)
        checkHIPErrors(hipblasSgemm(hipblasHandle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                                    ref_fc2.outputs, m_batchSize, ref_fc2.inputs,
                                    &alpha,
                                    pfc2, ref_fc2.inputs,
                                    fc1relu, ref_fc2.inputs,
                                    &beta,
                                    fc2, ref_fc2.outputs));
        // Add bias using GEMM's "beta" (fc2 += pfc2bias*1_vec')
        checkHIPErrors(hipblasSgemm(hipblasHandle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                    ref_fc2.outputs, m_batchSize, 1,
                                    &alpha,
                                    pfc2bias, ref_fc2.outputs,
                                    onevec, 1,
                                    &alpha,
                                    fc2, ref_fc2.outputs));

        // Softmax loss
        checkHIPDNN(hipdnnSoftmaxForward(hipdnnHandle, HIPDNN_SOFTMAX_ACCURATE, HIPDNN_SOFTMAX_MODE_CHANNEL,
                                       &alpha, fc2Tensor, fc2, &beta, fc2Tensor, result));
    }

    size_t SetBwdConvolutionTensors(hipdnnTensorDescriptor_t& srcTensorDesc, hipdnnTensorDescriptor_t& dstTensorDesc,
                                    hipdnnTensorDescriptor_t& filterDesc, hipdnnConvolutionDescriptor_t& convDesc,
                                    hipdnnConvolutionBwdFilterAlgo_t *falgo, hipdnnConvolutionBwdDataAlgo_t *dalgo)
    {
        size_t sizeInBytes = 0, tmpsize = 0;

        // If backprop filter algorithm was requested
        if (falgo)
        {
             checkHIPDNN(hipdnnGetConvolutionBackwardFilterAlgorithm(
                hipdnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
                HIPDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, falgo));

            checkHIPDNN(hipdnnGetConvolutionBackwardFilterWorkspaceSize(
                hipdnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
                *falgo, &tmpsize));

            sizeInBytes = std::max(sizeInBytes, tmpsize);
        }

        // If backprop data algorithm was requested
        if (dalgo)
        {
            checkHIPDNN(hipdnnGetConvolutionBackwardDataAlgorithm(
                hipdnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc,
                HIPDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, dalgo));

            checkHIPDNN(hipdnnGetConvolutionBackwardDataWorkspaceSize(
                hipdnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc,
                *dalgo, &tmpsize));

            sizeInBytes = std::max(sizeInBytes, tmpsize);
        }

        return sizeInBytes;
    }

    void Backpropagation(ConvBiasLayer& layer_conv1, MaxPoolLayer& layer_pool1, ConvBiasLayer& layer_conv2, MaxPoolLayer& layer_pool2,
                         float *data, float *labels, float *conv1, float *pool1, float *conv2, float *pool2, float *fc1, float *fc1relu,
                         float *fc2, float *fc2smax, float *dloss_data,
                         float *pconv1, float *pconv1bias,
                         float *pconv2, float *pconv2bias,
                         float *pfc1, float *pfc1bias,
                         float *pfc2, float *pfc2bias,
                         float *gconv1, float *gconv1bias, float *dpool1,
                         float *gconv2, float *gconv2bias, float *dconv2, float *dpool2,
                         float *gfc1, float *gfc1bias, float *dfc1, float *dfc1relu,
                         float *gfc2, float *gfc2bias, float *dfc2,
                         void *workspace, float *onevec)
    {
        float alpha = 1.0f, beta = 0.0f;

       float scalVal = 1.0f / static_cast<float>(m_batchSize);

        checkHIPErrors(hipSetDevice(m_gpuid));

        // Initialization (using the training error function)
        checkHIPErrors(hipMemcpyAsync(dloss_data, fc2smax, sizeof(float) * m_batchSize * ref_fc2.outputs, hipMemcpyDeviceToDevice));

        // Softmax layer
        // SoftmaxLossBackprop<<<RoundUp(m_batchSize, BW), BW>>>(labels, ref_fc2.outputs, m_batchSize, dloss_data);
        hipLaunchKernelGGL(SoftmaxLossBackprop,RoundUp(m_batchSize, BW), BW, 0, 0, labels, ref_fc2.outputs, m_batchSize, dloss_data);

        // Accounting for batch size in SGD
        checkHIPErrors(hipblasSscal(hipblasHandle, ref_fc2.outputs * m_batchSize, &scalVal, dloss_data, 1));

        // FC2 layer
        // Compute derivative with respect to weights: gfc2 = (fc1relu * dfc2smax')
        checkHIPErrors(hipblasSgemm(hipblasHandle, HIPBLAS_OP_N, HIPBLAS_OP_T, ref_fc2.inputs, ref_fc2.outputs, m_batchSize,
                                    &alpha, fc1relu, ref_fc2.inputs, dloss_data, ref_fc2.outputs, &beta, gfc2, ref_fc2.inputs));
        // Compute derivative with respect to bias: gfc2bias = dfc2smax * 1_vec
        checkHIPErrors(hipblasSgemv(hipblasHandle, HIPBLAS_OP_N, ref_fc2.outputs, m_batchSize,
                                    &alpha, dloss_data, ref_fc2.outputs, onevec, 1, &beta, gfc2bias, 1));
        // Compute derivative with respect to data (for previous layer): pfc2*dfc2smax (500x10*10xN)
        checkHIPErrors(hipblasSgemm(hipblasHandle, HIPBLAS_OP_N, HIPBLAS_OP_N, ref_fc2.inputs, m_batchSize, ref_fc2.outputs,
                                    &alpha, pfc2, ref_fc2.inputs, dloss_data, ref_fc2.outputs, &beta, dfc2, ref_fc2.inputs));

        // ReLU activation
        checkHIPDNN(hipdnnActivationBackward(hipdnnHandle, fc1Activation, &alpha,
                                           fc1Tensor, fc1relu, fc1Tensor, dfc2,
                                           fc1Tensor, fc1, &beta, fc1Tensor, dfc1relu));

        // FC1 layer
        // Compute derivative with respect to weights: gfc1 = (pool2 * dfc1relu')
        checkHIPErrors(hipblasSgemm(hipblasHandle, HIPBLAS_OP_N, HIPBLAS_OP_T, ref_fc1.inputs, ref_fc1.outputs, m_batchSize,
                                    &alpha, pool2, ref_fc1.inputs, dfc1relu, ref_fc1.outputs, &beta, gfc1, ref_fc1.inputs));
        // Compute derivative with respect to bias: gfc1bias = dfc1relu * 1_vec
        checkHIPErrors(hipblasSgemv(hipblasHandle, HIPBLAS_OP_N, ref_fc1.outputs, m_batchSize,
                                    &alpha, dfc1relu, ref_fc1.outputs, onevec, 1, &beta, gfc1bias, 1));
        // Compute derivative with respect to data (for previous layer): pfc1*dfc1relu (800x500*500xN)
        checkHIPErrors(hipblasSgemm(hipblasHandle, HIPBLAS_OP_N, HIPBLAS_OP_N, ref_fc1.inputs, m_batchSize, ref_fc1.outputs,
                                    &alpha, pfc1, ref_fc1.inputs, dfc1relu, ref_fc1.outputs, &beta, dfc1, ref_fc1.inputs));

        // Pool2 layer
        checkHIPDNN(hipdnnPoolingBackward(hipdnnHandle, poolDesc, &alpha,
                                        pool2Tensor, pool2, pool2Tensor, dfc1,
                                        conv2Tensor, conv2, &beta, conv2Tensor, dpool2));

        // Conv2 layer
        checkHIPDNN(hipdnnConvolutionBackwardBias(hipdnnHandle, &alpha, conv2Tensor,
                                                dpool2, &beta, conv2BiasTensor, gconv2bias));


        checkHIPDNN(hipdnnConvolutionBackwardFilter(hipdnnHandle, &alpha, pool1Tensor,
                                                  pool1, conv2Tensor, dpool2, conv2Desc,
                                                  conv2bwfalgo, workspace, m_workspaceSize,
                                                  &beta, conv2filterDesc, gconv2));

       checkHIPDNN(hipdnnConvolutionBackwardData(hipdnnHandle, &alpha, conv2filterDesc,
                                                pconv2, conv2Tensor, dpool2, conv2Desc, 
                                                conv2bwdalgo, workspace, m_workspaceSize,
                                                &beta, pool1Tensor, dconv2));

        // Pool1 layer
        checkHIPDNN(hipdnnPoolingBackward(hipdnnHandle, poolDesc, &alpha,
                                        pool1Tensor, pool1, pool1Tensor, dconv2,
                                        conv1Tensor, conv1, &beta, conv1Tensor, dpool1));

        // Conv1 layer
        checkHIPDNN(hipdnnConvolutionBackwardBias(hipdnnHandle, &alpha, conv1Tensor,
                                                dpool1, &beta, conv1BiasTensor, gconv1bias));

        checkHIPDNN(hipdnnConvolutionBackwardFilter(hipdnnHandle, &alpha, dataTensor,
                                                  data, conv1Tensor, dpool1, conv1Desc,
                                                  conv1bwfalgo, workspace, m_workspaceSize,
                                                  &beta, conv1filterDesc, gconv1));

        // No need for convBackwardData because there are no more layers below
    }

    void UpdateWeights(float learning_rate,
                       ConvBiasLayer& conv1, ConvBiasLayer& conv2,
                       float *pconv1, float *pconv1bias,
                       float *pconv2, float *pconv2bias,
                       float *pfc1, float *pfc1bias,
                       float *pfc2, float *pfc2bias,
                       float *gconv1, float *gconv1bias,
                       float *gconv2, float *gconv2bias,
                       float *gfc1, float *gfc1bias,
                       float *gfc2, float *gfc2bias)
    {
        float alpha = -learning_rate;

        checkHIPErrors(hipSetDevice(m_gpuid));

        // Conv1
        checkHIPErrors(hipblasSaxpy(hipblasHandle, static_cast<int>(conv1.pconv.size()),
                                    &alpha, gconv1, 1, pconv1, 1));
        checkHIPErrors(hipblasSaxpy(hipblasHandle, static_cast<int>(conv1.pbias.size()),
                                    &alpha, gconv1bias, 1, pconv1bias, 1));

        // Conv2
        checkHIPErrors(hipblasSaxpy(hipblasHandle, static_cast<int>(conv2.pconv.size()),
                                    &alpha, gconv2, 1, pconv2, 1));
        checkHIPErrors(hipblasSaxpy(hipblasHandle, static_cast<int>(conv2.pbias.size()),
                                    &alpha, gconv2bias, 1, pconv2bias, 1));

        // Fully connected 1
        checkHIPErrors(hipblasSaxpy(hipblasHandle, static_cast<int>(ref_fc1.pneurons.size()),
                                    &alpha, gfc1, 1, pfc1, 1));
        checkHIPErrors(hipblasSaxpy(hipblasHandle, static_cast<int>(ref_fc1.pbias.size()),
                                    &alpha, gfc1bias, 1, pfc1bias, 1));

        // Fully connected 2
        checkHIPErrors(hipblasSaxpy(hipblasHandle, static_cast<int>(ref_fc2.pneurons.size()),
                                    &alpha, gfc2, 1, pfc2, 1));
        checkHIPErrors(hipblasSaxpy(hipblasHandle, static_cast<int>(ref_fc2.pbias.size()),
                                    &alpha, gfc2bias, 1, pfc2bias, 1));
    }
};


///////////////////////////////////////////////////////////////////////////////////////////
// Main function

int main(int argc, char **argv)
{
#ifdef USE_GFLAGS
    gflags::ParseCommandLineFlags(&argc, &argv, true);
#endif

    size_t width, height, channels = 1;

    // Open input data
    printf("Reading input data\n");

    // Read dataset sizes
    size_t train_size = ReadUByteDataset(FLAGS_train_images.c_str(), FLAGS_train_labels.c_str(), nullptr, nullptr, width, height);
    size_t test_size = ReadUByteDataset(FLAGS_test_images.c_str(), FLAGS_test_labels.c_str(), nullptr, nullptr, width, height);
    if (train_size == 0)
        return 1;

    std::vector<uint8_t> train_images(train_size * width * height * channels), train_labels(train_size);
    std::vector<uint8_t> test_images(test_size * width * height * channels), test_labels(test_size);

    // Read data from datasets
    if (ReadUByteDataset(FLAGS_train_images.c_str(), FLAGS_train_labels.c_str(), &train_images[0], &train_labels[0], width, height) != train_size)
        return 2;
    if (ReadUByteDataset(FLAGS_test_images.c_str(), FLAGS_test_labels.c_str(), &test_images[0], &test_labels[0], width, height) != test_size)
        return 3;

    printf("Done. Training dataset size: %d, Test dataset size: %d\n", (int)train_size, (int)test_size);
    printf("Batch size: %lld, iterations: %d\n", FLAGS_batch_size, FLAGS_iterations);

    // This code snippet saves a random image and its label
    /*
    std::random_device rd_image;
    int random_image = rd_image() % train_size;
    std::stringstream ss; ss << "image-" << (int)train_labels[random_image] << ".pgm";
    SavePGMFile(&train_images[0] + random_image * width*height*channels, width, height, ss.str().c_str());
    */

    // Choose GPU
    int num_gpus;
    checkHIPErrors(hipGetDeviceCount(&num_gpus));
    if (FLAGS_gpu < 0 || FLAGS_gpu >= num_gpus)
    {
        printf("ERROR: Invalid GPU ID %d (There are %d GPUs on this machine)\n",
               FLAGS_gpu, num_gpus);
        return 4;
    }

    // Create the LeNet network architecture
    ConvBiasLayer conv1((int)channels, 20, 5, (int)width, (int)height);
    MaxPoolLayer pool1(2, 2);
    ConvBiasLayer conv2(conv1.out_channels, 50, 5, conv1.out_width / pool1.stride, conv1.out_height / pool1.stride);
    MaxPoolLayer pool2(2, 2);
    FullyConnectedLayer fc1((conv2.out_channels*conv2.out_width*conv2.out_height) / (pool2.stride * pool2.stride), 
                            500);
    FullyConnectedLayer fc2(fc1.outputs, 10);

    // Initialize HIPDNN/HIPBLAS training context
    TrainingContext context(FLAGS_gpu, FLAGS_batch_size, conv1, pool1, conv2, pool2, fc1, fc2);

    // Determine initial network structure
    bool bRet = true;
    if (FLAGS_pretrained)
    {
      bRet = conv1.FromFile("conv1");
      bRet &= conv2.FromFile("conv2");
      bRet &= fc1.FromFile("ip1");
      bRet &= fc2.FromFile("ip2");
    }
    if (!bRet || !FLAGS_pretrained)
    {
        // Create random network
        std::random_device rd;
        std::mt19937 gen(FLAGS_random_seed < 0 ? rd() : static_cast<unsigned int>(FLAGS_random_seed));

        // Xavier weight filling
        float wconv1 = sqrt(3.0f / (conv1.kernel_size * conv1.kernel_size * conv1.in_channels));
        std::uniform_real_distribution<> dconv1(-wconv1, wconv1);
        float wconv2 = sqrt(3.0f / (conv2.kernel_size * conv2.kernel_size * conv2.in_channels));
        std::uniform_real_distribution<> dconv2(-wconv2, wconv2);
        float wfc1 = sqrt(3.0f / (fc1.inputs * fc1.outputs));
        std::uniform_real_distribution<> dfc1(-wfc1, wfc1);
        float wfc2 = sqrt(3.0f / (fc2.inputs * fc2.outputs));
        std::uniform_real_distribution<> dfc2(-wfc2, wfc2);

        // Randomize network
        for (auto&& iter : conv1.pconv)
            iter = static_cast<float>(dconv1(gen));
        for (auto&& iter : conv1.pbias)
            iter = static_cast<float>(dconv1(gen));
        for (auto&& iter : conv2.pconv)
            iter = static_cast<float>(dconv2(gen));
        for (auto&& iter : conv2.pbias)
            iter = static_cast<float>(dconv2(gen));
        for (auto&& iter : fc1.pneurons)
            iter = static_cast<float>(dfc1(gen));
        for (auto&& iter : fc1.pbias)
            iter = static_cast<float>(dfc1(gen));
        for (auto&& iter : fc2.pneurons)
            iter = static_cast<float>(dfc2(gen));
        for (auto&& iter : fc2.pbias)
            iter = static_cast<float>(dfc2(gen));
    }

    /////////////////////////////////////////////////////////////////////////////
    // Create GPU data structures

    // Forward propagation data
    float *d_data, *d_labels, *d_conv1, *d_pool1, *d_conv2, *d_pool2, *d_fc1, *d_fc1relu, *d_fc2, *d_fc2smax;
    //                         Buffer    | Element       | N                   | C                  | H                                 | W
    //-----------------------------------------------------------------------------------------------------------------------------------------
    checkHIPErrors(hipMalloc(&d_data,    sizeof(float) * context.m_batchSize * channels           * height                            * width));
    checkHIPErrors(hipMalloc(&d_labels,  sizeof(float) * context.m_batchSize * 1                  * 1                                 * 1));
    checkHIPErrors(hipMalloc(&d_conv1,   sizeof(float) * context.m_batchSize * conv1.out_channels * conv1.out_height                  * conv1.out_width));
    checkHIPErrors(hipMalloc(&d_pool1,   sizeof(float) * context.m_batchSize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride)));
    checkHIPErrors(hipMalloc(&d_conv2,   sizeof(float) * context.m_batchSize * conv2.out_channels * conv2.out_height                  * conv2.out_width));
    checkHIPErrors(hipMalloc(&d_pool2,   sizeof(float) * context.m_batchSize * conv2.out_channels * (conv2.out_height / pool2.stride) * (conv2.out_width / pool2.stride)));
    checkHIPErrors(hipMalloc(&d_fc1,     sizeof(float) * context.m_batchSize * fc1.outputs));
    checkHIPErrors(hipMalloc(&d_fc1relu, sizeof(float) * context.m_batchSize * fc1.outputs));
    checkHIPErrors(hipMalloc(&d_fc2,     sizeof(float) * context.m_batchSize * fc2.outputs));
    checkHIPErrors(hipMalloc(&d_fc2smax, sizeof(float) * context.m_batchSize * fc2.outputs));

    // Network parameters
    float *d_pconv1, *d_pconv1bias, *d_pconv2, *d_pconv2bias;
    float *d_pfc1, *d_pfc1bias, *d_pfc2, *d_pfc2bias;

    checkHIPErrors(hipMalloc(&d_pconv1,     sizeof(float) * conv1.pconv.size()));
    checkHIPErrors(hipMalloc(&d_pconv1bias, sizeof(float) * conv1.pbias.size()));
    checkHIPErrors(hipMalloc(&d_pconv2,     sizeof(float) * conv2.pconv.size()));
    checkHIPErrors(hipMalloc(&d_pconv2bias, sizeof(float) * conv2.pbias.size()));
    checkHIPErrors(hipMalloc(&d_pfc1,       sizeof(float) * fc1.pneurons.size()));
    checkHIPErrors(hipMalloc(&d_pfc1bias,   sizeof(float) * fc1.pbias.size()));
    checkHIPErrors(hipMalloc(&d_pfc2,       sizeof(float) * fc2.pneurons.size()));
    checkHIPErrors(hipMalloc(&d_pfc2bias,   sizeof(float) * fc2.pbias.size()));

    // Network parameter gradients
    float *d_gconv1, *d_gconv1bias, *d_gconv2, *d_gconv2bias;
    float *d_gfc1, *d_gfc1bias, *d_gfc2, *d_gfc2bias;

    checkHIPErrors(hipMalloc(&d_gconv1,     sizeof(float) * conv1.pconv.size()));
    checkHIPErrors(hipMalloc(&d_gconv1bias, sizeof(float) * conv1.pbias.size()));
    checkHIPErrors(hipMalloc(&d_gconv2,     sizeof(float) * conv2.pconv.size()));
    checkHIPErrors(hipMalloc(&d_gconv2bias, sizeof(float) * conv2.pbias.size()));
    checkHIPErrors(hipMalloc(&d_gfc1,       sizeof(float) * fc1.pneurons.size()));
    checkHIPErrors(hipMalloc(&d_gfc1bias,   sizeof(float) * fc1.pbias.size()));
    checkHIPErrors(hipMalloc(&d_gfc2,       sizeof(float) * fc2.pneurons.size()));
    checkHIPErrors(hipMalloc(&d_gfc2bias,   sizeof(float) * fc2.pbias.size()));

    // Differentials w.r.t. data
    float *d_dpool1, *d_dpool2, *d_dconv2, *d_dfc1, *d_dfc1relu, *d_dfc2, *d_dfc2smax, *d_dlossdata;
    //                         Buffer     | Element       | N                   | C                  | H                                 | W
    //-----------------------------------------------------------------------------------------------------------------------------------------
    checkHIPErrors(hipMalloc(&d_dpool1,   sizeof(float) * context.m_batchSize * conv1.out_channels * conv1.out_height                  * conv1.out_width));
    checkHIPErrors(hipMalloc(&d_dpool2,   sizeof(float) * context.m_batchSize * conv2.out_channels * conv2.out_height                  * conv2.out_width));
    checkHIPErrors(hipMalloc(&d_dconv2,   sizeof(float) * context.m_batchSize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride)));
    checkHIPErrors(hipMalloc(&d_dfc1,     sizeof(float) * context.m_batchSize * fc1.inputs));
    checkHIPErrors(hipMalloc(&d_dfc1relu, sizeof(float) * context.m_batchSize * fc1.outputs));
    checkHIPErrors(hipMalloc(&d_dfc2,     sizeof(float) * context.m_batchSize * fc2.inputs));
    checkHIPErrors(hipMalloc(&d_dfc2smax, sizeof(float) * context.m_batchSize * fc2.outputs));
    checkHIPErrors(hipMalloc(&d_dlossdata,sizeof(float) * context.m_batchSize * fc2.outputs));

    // Temporary buffers and workspaces
    float *d_onevec;
    void *d_hipdnn_workspace = nullptr;
    checkHIPErrors(hipMalloc(&d_onevec, sizeof(float)* context.m_batchSize));
    if (context.m_workspaceSize > 0)
        checkHIPErrors(hipMalloc(&d_hipdnn_workspace, context.m_workspaceSize));

    /////////////////////////////////////////////////////////////////////////////

    // Copy initial network to device
    checkHIPErrors(hipMemcpyAsync(d_pconv1, &conv1.pconv[0],     sizeof(float) * conv1.pconv.size(),  hipMemcpyHostToDevice));
    checkHIPErrors(hipMemcpyAsync(d_pconv1bias, &conv1.pbias[0], sizeof(float) * conv1.pbias.size(),  hipMemcpyHostToDevice));
    checkHIPErrors(hipMemcpyAsync(d_pconv2, &conv2.pconv[0],     sizeof(float) * conv2.pconv.size(),  hipMemcpyHostToDevice));
    checkHIPErrors(hipMemcpyAsync(d_pconv2bias, &conv2.pbias[0], sizeof(float) * conv2.pbias.size(),  hipMemcpyHostToDevice));
    checkHIPErrors(hipMemcpyAsync(d_pfc1, &fc1.pneurons[0],      sizeof(float) * fc1.pneurons.size(), hipMemcpyHostToDevice));
    checkHIPErrors(hipMemcpyAsync(d_pfc1bias, &fc1.pbias[0],     sizeof(float) * fc1.pbias.size(),    hipMemcpyHostToDevice));
    checkHIPErrors(hipMemcpyAsync(d_pfc2, &fc2.pneurons[0],      sizeof(float) * fc2.pneurons.size(), hipMemcpyHostToDevice));
    checkHIPErrors(hipMemcpyAsync(d_pfc2bias, &fc2.pbias[0],     sizeof(float) * fc2.pbias.size(),    hipMemcpyHostToDevice));

    // Fill one-vector with ones
    //FillOnes<<<RoundUp(context.m_batchSize, BW), BW>>>(d_onevec, context.m_batchSize);
    hipLaunchKernelGGL(FillOnes,RoundUp(context.m_batchSize, BW), BW, 0, 0, d_onevec, context.m_batchSize);

    printf("Preparing dataset\n");

    // Normalize training set to be in [0,1]
    std::vector<float> train_images_float(train_images.size()), train_labels_float(train_size);
    for (size_t i = 0; i < train_size * channels * width * height; ++i)
        train_images_float[i] = (float)train_images[i] / 255.0f;

    for (size_t i = 0; i < train_size; ++i)
        train_labels_float[i] = (float)train_labels[i];

    printf("Training...\n");

    // Use SGD to train the network
    checkHIPErrors(hipDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < FLAGS_iterations; ++iter)
    {
        // Train
        int imageid = iter % (train_size / context.m_batchSize);

        // Prepare current batch on device
        checkHIPErrors(hipMemcpyAsync(d_data, &train_images_float[imageid * context.m_batchSize * width*height*channels],
                                        sizeof(float) * context.m_batchSize * channels * width * height, hipMemcpyHostToDevice));
        checkHIPErrors(hipMemcpyAsync(d_labels, &train_labels_float[imageid * context.m_batchSize],
                                        sizeof(float) * context.m_batchSize, hipMemcpyHostToDevice));

        // Forward propagation
       context.ForwardPropagation(d_data, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax,
                                   d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
                                   d_hipdnn_workspace, d_onevec);

        // Backward propagation
        context.Backpropagation(conv1, pool1, conv2, pool2,
                                d_data, d_labels, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax, d_dlossdata,
                                d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
                                d_gconv1, d_gconv1bias, d_dpool1, d_gconv2, d_gconv2bias, d_dconv2, d_dpool2, d_gfc1, d_gfc1bias, 
                                d_dfc1, d_dfc1relu, d_gfc2, d_gfc2bias, d_dfc2, d_hipdnn_workspace, d_onevec);

        // Compute learning rate
        float learningRate = static_cast<float>(FLAGS_learning_rate * pow((1.0 + FLAGS_lr_gamma * iter), (-FLAGS_lr_power)));

        // Update weights
        context.UpdateWeights(learningRate, conv1, conv2,
                              d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
                              d_gconv1, d_gconv1bias, d_gconv2, d_gconv2bias, d_gfc1, d_gfc1bias, d_gfc2, d_gfc2bias);
    }
    checkHIPErrors(hipDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    printf("Iteration time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / FLAGS_iterations);

    if (FLAGS_save_data)
    {
        // Copy trained weights from GPU to CPU
        checkHIPErrors(hipMemcpy(&conv1.pconv[0], d_pconv1, sizeof(float) * conv1.pconv.size(), hipMemcpyDeviceToHost));
        checkHIPErrors(hipMemcpy(&conv1.pbias[0], d_pconv1bias, sizeof(float) * conv1.pbias.size(), hipMemcpyDeviceToHost));
        checkHIPErrors(hipMemcpy(&conv2.pconv[0], d_pconv2, sizeof(float) * conv2.pconv.size(), hipMemcpyDeviceToHost));
        checkHIPErrors(hipMemcpy(&conv2.pbias[0], d_pconv2bias, sizeof(float) * conv2.pbias.size(), hipMemcpyDeviceToHost));
        checkHIPErrors(hipMemcpy(&fc1.pneurons[0], d_pfc1, sizeof(float) * fc1.pneurons.size(), hipMemcpyDeviceToHost));
        checkHIPErrors(hipMemcpy(&fc1.pbias[0], d_pfc1bias, sizeof(float) * fc1.pbias.size(), hipMemcpyDeviceToHost));
        checkHIPErrors(hipMemcpy(&fc2.pneurons[0], d_pfc2, sizeof(float) * fc2.pneurons.size(), hipMemcpyDeviceToHost));
        checkHIPErrors(hipMemcpy(&fc2.pbias[0], d_pfc2bias, sizeof(float) * fc2.pbias.size(), hipMemcpyDeviceToHost));

        // Now save data
        printf("Saving data to file\n");
        conv1.ToFile("conv1");
        conv2.ToFile("conv2");
        fc1.ToFile("ip1");
        fc2.ToFile("ip2");
    }


    float classification_error = 1.0f;

    int classifications = FLAGS_classify;
    if (classifications < 0)
        classifications = (int)test_size;

    // Test the resulting neural network's classification
    if (classifications > 0)
    {
        // Initialize a TrainingContext structure for testing (different batch size)
        TrainingContext test_context(FLAGS_gpu, 1, conv1, pool1, conv2, pool2, fc1, fc2);

        // Ensure correct workspaceSize is allocated for testing
        if (context.m_workspaceSize < test_context.m_workspaceSize)
        {
            checkHIPErrors(hipFree(d_hipdnn_workspace));
            checkHIPErrors(hipMalloc(&d_hipdnn_workspace, test_context.m_workspaceSize));
        }

        int num_errors = 0;
        for (int i = 0; i < classifications; ++i)
        {
            std::vector<float> data(width * height);
            // Normalize image to be in [0,1]
            for (int j = 0; j < width * height; ++j)
                data[j] = (float)test_images[i * width*height*channels + j] / 255.0f;

            checkHIPErrors(hipMemcpyAsync(d_data, &data[0], sizeof(float) * width * height, hipMemcpyHostToDevice));

            // Forward propagate test image
            test_context.ForwardPropagation(d_data, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax,
                                            d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias,
                                            d_pfc2, d_pfc2bias, d_hipdnn_workspace, d_onevec);

            // Perform classification
            std::vector<float> class_vec(10);

            // Copy back result
            checkHIPErrors(hipMemcpy(&class_vec[0], d_fc2smax, sizeof(float) * 10, hipMemcpyDeviceToHost));

            // Determine classification according to maximal response
            int chosen = 0;
            for (int id = 1; id < 10; ++id)
            {
                if (class_vec[chosen] < class_vec[id]) chosen = id;
            }

            if (chosen != test_labels[i])
                ++num_errors;
        }
        classification_error = (float)num_errors / (float)classifications;

        printf("Classification result: %.2f%% error (used %d images)\n", classification_error * 100.0f, (int)classifications);
    }

    // Free data structures
    checkHIPErrors(hipFree(d_data));
    checkHIPErrors(hipFree(d_conv1));
    checkHIPErrors(hipFree(d_pool1));
    checkHIPErrors(hipFree(d_conv2));
    checkHIPErrors(hipFree(d_pool2));
    checkHIPErrors(hipFree(d_fc1));
    checkHIPErrors(hipFree(d_fc2));
    checkHIPErrors(hipFree(d_pconv1));
    checkHIPErrors(hipFree(d_pconv1bias));
    checkHIPErrors(hipFree(d_pconv2));
    checkHIPErrors(hipFree(d_pconv2bias));
    checkHIPErrors(hipFree(d_pfc1));
    checkHIPErrors(hipFree(d_pfc1bias));
    checkHIPErrors(hipFree(d_pfc2));
    checkHIPErrors(hipFree(d_pfc2bias));
    checkHIPErrors(hipFree(d_gconv1));
    checkHIPErrors(hipFree(d_gconv1bias));
    checkHIPErrors(hipFree(d_gconv2));
    checkHIPErrors(hipFree(d_gconv2bias));
    checkHIPErrors(hipFree(d_gfc1));
    checkHIPErrors(hipFree(d_gfc1bias));
    checkHIPErrors(hipFree(d_dfc1));
    checkHIPErrors(hipFree(d_gfc2));
    checkHIPErrors(hipFree(d_gfc2bias));
    checkHIPErrors(hipFree(d_dfc2));
    checkHIPErrors(hipFree(d_dpool1));
    checkHIPErrors(hipFree(d_dconv2));
    checkHIPErrors(hipFree(d_dpool2));
    checkHIPErrors(hipFree(d_labels));
    checkHIPErrors(hipFree(d_dlossdata));
    checkHIPErrors(hipFree(d_onevec));
    if (d_hipdnn_workspace != nullptr)
        checkHIPErrors(hipFree(d_hipdnn_workspace));

    return 0;
}
