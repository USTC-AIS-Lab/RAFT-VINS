#include <iostream>
#include <string>
#include <memory>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <torch/torch.h>

#include "logging.h"
#include "logger.h"
// #include "src/commonold.h"
#include "common.h"

using namespace std;

// struct InferDeleter{
//     template <typename T>
//     void operator()(T* obj) const{
//         if (obj)
//             obj->destroy();
//     }
// };
bool MyBuild(const string &onnx_path, const string &tensorrt_path, const int& flag) {
    //创建builder
    cout << "create Infer Builder" << endl;
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if(!builder){
        return false;
    }

    //创建网络定义
    cout << "create Network" << endl;
    const auto explictBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explictBatch));
    if(!network) {
        return false;
    }

    //创建config
    cout << "create BuilderConfig" << endl;
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if(!config) {
        return false;
    }

    //创建parse
    cout << "create Parse" << endl;
    auto parse = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if(!parse){
        return false;
    }
 
    //读取模型文件
    auto verbosity = sample::gLogger.getReportableSeverity();
    auto parsed = parse->parseFromFile(onnx_path.c_str(), static_cast<int>(verbosity));
    if(!parsed){
        return false;
    }
     
    //设置工作层空间大小
    config->setMaxWorkspaceSize(1 << 20);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    //设置DLA
    const int useDLACore = -1; //消费级GPU没有DLA
    samplesCommon::enableDLA(builder.get(), config.get(), useDLACore);

    //创建不同分辨率
    if (flag == 0) {
        auto profile_0 = builder->createOptimizationProfile();
        profile_0->setDimensions("img0", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, 100, 100});
        profile_0->setDimensions("img0", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, 3, 480, 640});
        profile_0->setDimensions("img0", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, 3, 800, 1600});
        profile_0->setDimensions("img1", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, 100, 100});
        profile_0->setDimensions("img1", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, 3, 480, 640});
        profile_0->setDimensions("img1", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, 3, 800, 1600});
        config->addOptimizationProfile(profile_0);
        cout << "input shape: " << network->getInput(0)->getName() << " " << network->getInput(0)->getDimensions() << endl;
        cout << "input shape: " << network->getInput(1)->getName() << " " << network->getInput(1)->getDimensions() << endl;
        cout << "out shape: " << network->getOutput(0)->getName() << " " << network->getOutput(0)->getDimensions() << endl;
        cout << "out shape: " << network->getOutput(1)->getName() << " " << network->getOutput(1)->getDimensions() << endl;
    } else if (flag == 1) {
        auto profile_0 = builder->createOptimizationProfile();
        profile_0->setDimensions("img0", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, 100, 100});
        profile_0->setDimensions("img0", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, 3, 480, 640});
        profile_0->setDimensions("img0", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, 3, 800, 1600});
        config->addOptimizationProfile(profile_0);
        cout << "input shape: " << network->getInput(0)->getName() << " " << network->getInput(0)->getDimensions() << endl;
        cout << "out shape: " << network->getOutput(0)->getName() << " " << network->getOutput(0)->getDimensions() << endl;
    } else if(flag == 2) {
        auto profile_0 = builder->createOptimizationProfile();
        profile_0->setDimensions("net", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 96, 10, 20));
        profile_0->setDimensions("net", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 96, 60, 80));
        profile_0->setDimensions("net", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 96, 100, 200));
        profile_0->setDimensions("inp", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 64, 10, 20));
        profile_0->setDimensions("inp", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 64, 60, 80));
        profile_0->setDimensions("inp", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 64, 100, 200));
        profile_0->setDimensions("corr", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 196, 10, 20));
        profile_0->setDimensions("corr", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 196, 60, 80));
        profile_0->setDimensions("corr", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 196, 100, 200));
        profile_0->setDimensions("flow", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 2, 10, 20));
        profile_0->setDimensions("flow", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 2, 60, 80));
        profile_0->setDimensions("flow", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 2, 100, 200));
        config->addOptimizationProfile(profile_0);
        cout << "input shape: " << network->getInput(0)->getName() << " " << network->getInput(0)->getDimensions() << endl;
        cout << "input shape: " << network->getInput(1)->getName() << " " << network->getInput(1)->getDimensions() << endl;
        cout << "input shape: " << network->getInput(2)->getName() << " " << network->getInput(2)->getDimensions() << endl;
        cout << "input shape: " << network->getInput(3)->getName() << " " << network->getInput(3)->getDimensions() << endl;
        cout << "out shape: " << network->getOutput(0)->getName() << " " << network->getOutput(0)->getDimensions() << endl;
        cout << "out shape: " << network->getOutput(1)->getName() << " " << network->getOutput(1)->getDimensions() << endl;
        cout << "out shape: " << network->getOutput(2)->getName() << " " << network->getOutput(2)->getDimensions() << endl;
    }
    //构建engine
    // cout << "build Engine with Config" << endl;
    // std::shared_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    // if(!plan) {
    //     return false;
    // }
    
    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    if(!engine) {
        return false;
    }

    //序列化模型
    cout << "serializeModel" << endl;
    auto serializeModel = engine->serialize(); //nvinfer1::IHostMemory

    // //将序列化模型拷贝到字符串
    // std::string serialize_str;

    // serialize_str.resize(serializeModel->size());
    // memcpy((void*)serialize_str.data(), serializeModel->data(), serializeModel->size());
    // //将字符串输出到文件中
    // std::ofstream serialize_stream(tensorrt_path, std::ios::binary);
    // serialize_stream << serialize_str;
    // serialize_stream.close();
    // cout << "done" << endl;
    std::ofstream p(tensorrt_path, std::ios::binary); 
    if (!p) { 
            std::cerr << "could not open output file to save model" << std::endl; 
            return -1; 
    } 
    p.write(reinterpret_cast<const char*>(serializeModel->data()), serializeModel->size()); 
    p.close();
    std::cout << "generating file done!" << std::endl; 

    // Release resources 
    // serializeModel->destroy(); 
    // network->destroy(); 
    // engine->destroy(); 
    // config->destroy(); 
    // builder->destroy(); 
    return true;
}

int main (int argc, char** argv) {
    MyBuild("/home/zj/workspace/paper_final/RAFT_CPP_v10/onnx_models/small_mask/orin_fnet-small_mask.onnx", \
                "/home/zj/workspace/paper_final/RAFT_CPP_v10/onnx_models/small_mask/orin_fnet-small_mask.engine", 0);
    MyBuild("/home/zj/workspace/paper_final/RAFT_CPP_v10/onnx_models/small_mask/orin_cnet-small_mask.onnx", \
                "/home/zj/workspace/paper_final/RAFT_CPP_v10/onnx_models/small_mask/orin_cnet-small_mask.engine", 1);
    MyBuild("/home/zj/workspace/paper_final/RAFT_CPP_v10/onnx_models/small_mask/orin_update-small_mask.onnx", \
                "/home/zj/workspace/paper_final/RAFT_CPP_v10/onnx_models/small_mask/orin_update-small_mask.engine", 2);
}

// int main(){
//     std::string path = "output.txt";
//     std::ofstream outfile(path, std::ios::binary);
//     outfile << 1231231 << endl;
//     outfile << "12312313123" << endl;
// }


// int main() {
//     // 创建一个张量
//     torch::Tensor tensor = torch::tensor({3.0, 3.0});

//     // 使用 .data_ptr 访问张量的数据指针
//     float* data_ptr = tensor.data_ptr<float>();

//     // 打印数据的前几个元素
//     for (int i = 0; i < 1; i++) {
//         std::cout << data_ptr[i] << " ";
//     }
//     std::cout << std::endl;

//     return 0;
// }

