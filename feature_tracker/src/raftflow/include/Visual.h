/*******************************************************
 *
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of RAFT_Libtorch.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <opencv2/opencv.hpp>
#include <torch/torch.h>


torch::Tensor flow_to_image(torch::Tensor &flow_uv);
cv::Mat visual_flow_image(torch::Tensor &img,torch::Tensor &flow_uv);
cv::Mat visual_flow_image(torch::Tensor &flow_uv);

