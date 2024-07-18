/*******************************************************
 *
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of RAFT_Libtorch.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "Pipeline.h"

namespace F = torch::nn::functional;


torch::Tensor Pipeline::process(cv::Mat &img)
{
    // TicToc tt;
    int h = img.rows;
    int w = img.cols;
    int channel = img.channels();
    torch::Tensor input_tensor;
    if (channel == 1) {
        // std::cout << "****************channel == 1*****************" << std::endl;
        cv::Mat img_float;
        img.convertTo(img_float,CV_32FC1);
        auto input_tensor_one = torch::from_blob(img_float.data, {h, w ,1}, torch::kFloat32);
        // std::cout << "input_tensor_one: " << input_tensor_one.sizes() << std::endl;
        input_tensor = torch::cat({input_tensor_one, input_tensor_one, input_tensor_one}, 2).to(torch::kCUDA);
        // std::cout << "input_tensor: " << input_tensor.sizes() << std::endl;
        // exit(0);
        // cv::cvtColor(img, color_image, cv::COLOR_GRAY2BGR);
        // img = color_image.clone();
    } else {
        // std::cout << "img1.channels() " << img.channels() << std::endl; 
        cv::Mat img_float;
        img.convertTo(img_float,CV_32FC3);
        input_tensor = torch::from_blob(img_float.data, {h,w ,3 }, torch::kFloat32).to(torch::kCUDA);
    }
    // auto input_tensor = torch::from_blob(img_float.data, {h,w ,3 }, torch::kFloat32).to(torch::kCPU);

    ///预处理
    input_tensor = 2*(input_tensor/255.0f) - 1.0f;

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
    int h_pad = ((int(h/8)+1)*8 - h)%8;
    int w_pad = ((int(w/8)+1)*8 - w)%8;

    //前两个数pad是2维度，中间两个数pad第1维度，后两个数pad 第0维度
    input_tensor = F::pad(input_tensor, F::PadFuncOptions({w_pad, 0, h_pad, 0, 0, 0}).mode(torch::kConstant));


    return input_tensor.unsqueeze(0).contiguous();
}
