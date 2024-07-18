#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <memory>

using Tensor = torch::Tensor;
using std::vector;

class Torch_script {
  public:
    Torch_script();
    vector<Tensor> Forward(Tensor &tensor0, Tensor &tensor1);

  private:
    std::shared_ptr<torch::jit::Module> raft;
};

class Raft_torch {
  public:
    Raft_torch();

    std::unique_ptr<torch::jit::Module> fnet_;
    std::unique_ptr<torch::jit::Module> cnet_;
    std::unique_ptr<torch::jit::Module> decoder_;
};