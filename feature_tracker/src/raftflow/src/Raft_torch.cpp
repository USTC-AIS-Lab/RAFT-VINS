#include "Raft_torch.h"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <memory>

using namespace std;

Torch_script::Torch_script() {
    raft = std::make_unique<torch::jit::Module>(torch::jit::load(\
                            "/home/zj/workspace/slambook2/ch13_v2/src/raftflow/script_models/kitti.pt"));
}

vector<Tensor> Torch_script::Forward(Tensor &tensor0, Tensor &tensor1) {
    auto result = raft->forward({tensor0, tensor1}).toTensorVector();
    return result;
}

Trace_script::Trace_script() {
    fnet_ = std::make_unique<torch::jit::Module>(
                    torch::jit::load("/home/zj/workspace/slambook2/ch13_v2/src/raftflow/script_models/fnet.pt"));
    cnet_ = std::make_unique<torch::jit::Module>(
                    torch::jit::load("/home/zj/workspace/slambook2/ch13_v2/src/raftflow/script_models/cnet.pt"));
    decoder_ = std::make_unique<torch::jit::Module>(
                    torch::jit::load("/home/zj/workspace/slambook2/ch13_v2/src/raftflow/script_models/decoder.pt"));
}
