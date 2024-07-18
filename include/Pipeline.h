#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

class Pipeline {
  public:
    // torch::Tensor process(cv::Mat img);
    // torch::Tensor myprocess(cv::Mat img);
    torch::Tensor process(cv::Mat& img);


};