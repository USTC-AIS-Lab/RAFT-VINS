/**************************************************************88****
 *
 * Copyright (C) 2022, Zhang Jie, USTC
 *
 * This file is modified of RAFT_Libtorch by Chen Jiaqu.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************************/
#include "Pipeline.h"

namespace F = torch::nn::functional;

torch::Tensor Pipeline::process(cv::Mat &img)
{
    // TicToc tt;
    int h = img.rows;
    int w = img.cols;
    int channel = img.channels();
    torch::Tensor input_tensor;
    cv::Mat img_float;
    if (channel == 1) {
        img.convertTo(img_float,CV_32FC1);
        auto input_tensor_one = torch::from_blob(img_float.data, {h, w ,1}, torch::kFloat32);
        input_tensor = torch::cat({input_tensor_one, input_tensor_one, input_tensor_one}, 2).to(torch::kCUDA);
    } else {
        img.convertTo(img_float,CV_32FC3);
        input_tensor = torch::from_blob(img_float.data, {h, w, 3}, torch::kFloat32).to(torch::kCUDA);
    }

    ///预处理
    input_tensor = 2 * (input_tensor / 255.0f) - 1.0f;

    ///bgr->rgb
    input_tensor = torch::cat({
        input_tensor.index({"...",2}).unsqueeze(2),
        input_tensor.index({"...",1}).unsqueeze(2),
        input_tensor.index({"...",0}).unsqueeze(2)
        },2);
    // Debugs("setInputTensorCuda bgr->rgb:{} {} ms", DimsToStr(input_tensor.sizes()), tt.TocThenTic());

    ///hwc->chw
    input_tensor = input_tensor.permute({2,0,1});
    // Debugs("setInputTensorCuda hwc->chw:{} {} ms", DimsToStr(input_tensor.sizes()), tt.TocThenTic());

    ///pad
    int pad_wd = ((int(w / 8) + 1) * 8 - w) % 8;
    int pad_ht = ((int(h / 8) + 1) * 8 - h) % 8;
    
    //前两个数pad是2维度，中间两个数pad第1维度，后两个数pad 第0维度
    // input_tensor = F::pad(input_tensor, F::PadFuncOptions({w_pad, 0, h_pad, 0, 0, 0}).mode(torch::kConstant));
    F::PadFuncOptions pad_tensor({(pad_wd / 2), pad_wd - (pad_wd / 2), (pad_ht / 2), pad_ht - (pad_ht / 2)});
    input_tensor = torch::nn::functional::pad(input_tensor, pad_tensor.mode(torch::kReplicate));

    return input_tensor.unsqueeze(0).contiguous();
}