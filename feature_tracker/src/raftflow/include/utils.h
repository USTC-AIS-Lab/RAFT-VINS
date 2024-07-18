#pragma once

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

using Tensor = torch::Tensor;

class Tictoc {
  public:
    Tictoc(){}

    void tic() {
        start_ = std::chrono::system_clock::now();
        // std::cout << start_ << std::endl;
    }

    double toc() {
        end_ = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_time = end_ - start_;
        return elapsed_time.count() * 1000;
    }

    double TocThenTic() {
        auto t = toc();
        tic();
        return t;
    }

    void TocprintTic(const char* str) {
        std::cout << str << ": " << toc() << " ms" << std::endl;
        tic();
    }

  private:
    std::chrono::time_point<std::chrono::system_clock> start_, end_;

};

inline Tensor bilinear_sampler(Tensor &img, Tensor &coords) {
    int high = img.sizes()[2];
    int wide = img.sizes()[3];
    Tensor xgrid = coords.split({1, 1}, -1)[0];
    // cout << "xgrid.sizes()" << xgrid.sizes() << endl; //xgrid.sizes()[7332, 9, 9, 1] 47*156
    // cout << xgrid << endl;
    Tensor ygrid = coords.split({1, 1}, -1)[1];

    //调整到【-1 ，1】
    xgrid = 2 * xgrid / (wide - 1) - 1;
    ygrid = 2 * ygrid / (high - 1) - 1;

    Tensor grid = torch::cat({xgrid, ygrid}, -1);
    auto opt = torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true);
    img = torch::nn::functional::grid_sample(img, grid, opt);

    return img;
}
