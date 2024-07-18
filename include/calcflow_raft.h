#pragma once

#include <tuple>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "Raft_torch.h"
#include "Pipeline.h"
#include "utils.h"

#include "common.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

using namespace std;

class calcflow_raft {
  public:
    calcflow_raft();
    void raft_predict(cv::Mat &img1, cv::Mat &img2, std::vector<cv::Point2f>&kps_last, \
                                  std::vector<cv::Point2f>&kps_curr, std::vector<uchar>&status);
    void Corr(torch::Tensor tensor1, torch::Tensor tensor2);
    Tensor Index_correlation(Tensor coords, bool flag);
    Tensor upsample_flow(Tensor flow, Tensor mask);
    Tensor upflow8(Tensor flow);
    tuple<torch::Tensor, torch::Tensor> Initialize(torch::Tensor& tensor1);
    Tensor trace_predict(Tensor& img1_tensor, Tensor& img2_tensor);
    tuple<Tensor, Tensor> fnet_forward(Tensor& tensor0, Tensor& tensor1);
    Tensor cnet_forward(Tensor& tensor0);
    tuple<Tensor, Tensor, Tensor> decoder_forward(Tensor& tensor0, Tensor& tensor1, Tensor& tensor2, Tensor& tensor3);

  private:
    std::unique_ptr<Torch_script> raft_pred;
    std::unique_ptr<Pipeline> convert;
    std::vector<Tensor> corr_pyramid; // 直接初始化

    std::shared_ptr<nvinfer1::IRuntime> fnet_runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> fnet_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> fnet_context;
    
    std::shared_ptr<nvinfer1::IRuntime> cnet_runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> cnet_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> cnet_context;

    std::shared_ptr<nvinfer1::IRuntime> decoder_runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> decoder_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> decoder_context;

    cudaStream_t stream{};

    Tictoc trace_time;
    Tictoc sys_time;
    Tictoc time_;
};

