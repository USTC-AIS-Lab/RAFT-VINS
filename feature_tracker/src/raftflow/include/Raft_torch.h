#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <memory>

using Tensor = torch::Tensor;

class Torch_script {
  public:
    Torch_script();
    std::vector<Tensor> Forward(Tensor &tensor0, Tensor &tensor1);

  private:
    std::unique_ptr<torch::jit::Module> raft;
};

class Trace_script {
  public:
    Trace_script();

    std::unique_ptr<torch::jit::Module> fnet_;
    std::unique_ptr<torch::jit::Module> cnet_;
    std::unique_ptr<torch::jit::Module> decoder_;
};
