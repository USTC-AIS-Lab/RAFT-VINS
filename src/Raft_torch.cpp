#include "Raft_torch.h"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <memory>

Torch_script::Torch_script() {
    // std::cout << "load model" << std::endl;
    raft = std::make_unique<torch::jit::Module>(torch::jit::load("/home/zj/workspace/RAFT_CPP/RAFT_CPP_v1/script_models/mythings3d.pt"));
    // raft->to(torch::kCPU);
}

vector<Tensor> Torch_script::Forward(Tensor &tensor0, Tensor &tensor1) {
    auto result = raft->forward({tensor0, tensor1}).toTensorVector();
    return result;
}

Raft_torch::Raft_torch() {
    fnet_ = std::make_unique<torch::jit::Module>(torch::jit::load("/home/zj/workspace/RAFT_CPP/RAFT_CPP_v3/script_models/mynewerfnet.pt"));
    cnet_ = std::make_unique<torch::jit::Module>(torch::jit::load("/home/zj/workspace/RAFT_CPP/RAFT_CPP_v3/script_models/newerer_cnet.pt"));
    decoder_ = std::make_unique<torch::jit::Module>(torch::jit::load("/home/zj/workspace/RAFT_CPP/RAFT_CPP_v3/script_models/newdecoder.pt"));
}


