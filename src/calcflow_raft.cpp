#include "calcflow_raft.h"
#include "Visual.h"

namespace F = torch::nn::functional;

bool CreateModel(std::shared_ptr<nvinfer1::IRuntime> &runtime,
                    std::shared_ptr<nvinfer1::ICudaEngine> &engine,
                    std::unique_ptr<nvinfer1::IExecutionContext> &context,
                    const string &path) {
    char *trtModelStream = nullptr; 
    size_t size{ 0 }; 

    std::ifstream file(path, std::ios::binary); 
    if (file.good()) { 
            file.seekg(0, file.end); 
            size = file.tellg(); 
            file.seekg(0, file.beg); 
            trtModelStream = new char[size]; 
            assert(trtModelStream); 
            file.read(trtModelStream, size); 
            file.close(); 
    } 
	nvinfer1::ILogger* gLogger;
    bool didInitPlugins = initLibNvInferPlugins(gLogger, ""); 
    
    //创建runtime
    std::cout << "create inferruntime" << endl;
    runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if(!runtime){
        std::cout << "can not create runtime" << std::endl; 
    }

    //反序列化模型
    std::cout << "create engine deserializeCudaEngine" << endl;
    engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(trtModelStream, size), 
                        samplesCommon::InferDeleter());
    if(!engine){
        std::cout << "can not create engine" << std::endl; 
    }

    //创建执行上下文
    std::cout << "create ExecutionContext" << endl;
    context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if(!context) {
        std::cout << "can not create context" << std::endl; 
    }
    return true;
}

calcflow_raft::calcflow_raft() {
    raft_pred = std::make_unique<Torch_script>();
    convert = std::make_unique<Pipeline>();
    bool result_fnet = CreateModel(fnet_runtime, fnet_engine, fnet_context, 
    "/home/zj/workspace/paper_final/vins_ws_v4/src/VINS-Mono/feature_tracker/src/raftflow/engine/orin_fnet-small_mask.engine");
    bool result_cnet = CreateModel(cnet_runtime, cnet_engine, cnet_context, 
    "/home/zj/workspace/paper_final/vins_ws_v4/src/VINS-Mono/feature_tracker/src/raftflow/engine/orin_cnet-small_mask.engine");
    bool result_decoder = CreateModel(decoder_runtime, decoder_engine, decoder_context, 
    "/home/zj/workspace/paper_final/vins_ws_v4/src/VINS-Mono/feature_tracker/src/raftflow/engine/orin_update-small_mask.engine");
}

void calcflow_raft::Corr(torch::Tensor tensor1, torch::Tensor tensor2) {
    int batch = tensor1.sizes()[0];
    int dim = tensor1.sizes()[1];
    int ht = tensor1.sizes()[2];
    int wd = tensor1.sizes()[3];

    tensor1 = tensor1.view({batch, dim, ht*wd}); //[1, 256, 47 * 156]
    tensor2 = tensor2.view({batch, dim, ht*wd}); //[1, 256, 47 * 156]

    Tensor corr = torch::matmul(tensor1.transpose(1, 2), tensor2); //corr.sizes()_after: [1, 7332, 7332] [1， 47 * 156, 47 * 156]
    corr = corr.view({batch, ht, wd, 1, ht, wd}); //corr.sizes()_after: [1, 7332, 7332]
    corr = corr / torch::sqrt(torch::tensor(dim).to(torch::kFloat32)); //corr.sizes()_later: [1, 47, 156, 1, 47, 156]

    corr = corr.reshape({batch*ht*wd, 1, ht, wd});  // corr.sizes()[7332, 1, 47, 156]

    //池化，构建金字塔
    corr_pyramid.clear();
    corr_pyramid.push_back(corr);
    static auto opt = F::AvgPool2dFuncOptions(2).stride(2);
    for (int i = 0; i < 3; i++) {
        corr = F::avg_pool2d(corr, opt);
        corr_pyramid.push_back(corr); 
    }
}

Tensor calcflow_raft::Index_correlation(Tensor coords, bool flag) {
    int radius;
    if (flag)
        radius = 3;
    else 
        radius = 4;
    coords = coords.permute({0, 2, 3, 1});
    int batch = coords.sizes()[0];
    int ht = coords.sizes()[1];
    int wd = coords.sizes()[2];
    vector<Tensor> out_pyramid;
    auto opt = torch::TensorOptions(torch::kCUDA);
    Tensor dx = torch::linspace(-radius, radius, 2 * radius + 1, opt);
    Tensor dy = torch::linspace(-radius, radius, 2 * radius + 1, opt);
    Tensor delta = torch::stack(torch::meshgrid({dx, dy}), -1); //0 : delta.szies()[2, 9, 9] -1 : [9, 9, 2]
    Tensor delta_lvl = delta.view({1, 2 * radius + 1, 2 * radius + 1, 2});
    for (int i = 0; i < 4; i++) {
        Tensor corr = corr_pyramid[i]; //[B,C,H,W]
        Tensor centroid_lvl = coords.reshape({batch*ht*wd, 1, 1, 2}) / pow(2, i);
        Tensor coords_lvl = centroid_lvl + delta_lvl; //[7040, 9, 9, 2]

        corr = bilinear_sampler(corr, coords_lvl); //corr.sizes()[7332, 1, 9, 9]
        corr = corr.view({batch, ht, wd, -1}); //[1, 47, 156, 81]
        out_pyramid.push_back(corr);
    }
    Tensor out = torch::cat(out_pyramid, -1); //[1, 47, 156, 324]
    out = out.permute({0, 3, 1, 2}).contiguous().to(torch::kFloat32);
    return out;
}

Tensor calcflow_raft::upsample_flow(Tensor flow, Tensor mask) {
    int batch = flow.sizes()[0];
    int ht = flow.sizes()[2];
    int wd = flow.sizes()[3];
    mask = mask.view({batch, 1, 9, 8, 8, ht, wd}); //mask.sizes()[1, 1, 9, 8, 8, 47, 156]
    mask = torch::softmax(mask, 2); 
    auto opt = F::UnfoldFuncOptions({3, 3}).padding(1); 
    Tensor up_flow = F::unfold(8 * flow, opt); //flow.sizes()[1, 2, 47, 156] up_flow.sizes()[1, 18, 7332]
    up_flow = up_flow.view({batch, 2, 9, 1, 1, ht, wd});

    up_flow = torch::sum(mask * up_flow, 2); //up_flow.sizes()[1, 2, 8, 8, 47, 156]
    up_flow = up_flow.permute({0, 1, 4, 2, 5, 3});
    up_flow = up_flow.reshape({batch, 2, 8*ht, 8*wd});
    return up_flow;
}

Tensor calcflow_raft::upflow8(Tensor flow) {
    // Tensor new_size = (8 * flow.sizes()[2], 8 * flow.sizes()[3])
    auto opt = F::InterpolateFuncOptions().size(std::vector<int64_t>({8 * flow.sizes()[2], 8 * flow.sizes()[3]})).mode(torch::kBilinear).align_corners(true);
    return 8 * F::interpolate(flow, opt);
}

tuple<torch::Tensor, torch::Tensor> calcflow_raft::Initialize(torch::Tensor& tensor1) {
    int batch = tensor1.sizes()[0];
    int ht = tensor1.sizes()[2];
    int wd = tensor1.sizes()[3];

    auto opt = torch::TensorOptions(torch::kCUDA);
    vector<Tensor> coords = torch::meshgrid({torch::arange(ht, opt), torch::arange(wd, opt)});
    //coords.size() 2，  coords[0].sizes() [47, 156]
    Tensor coords_0 = torch::stack({coords[1], coords[0]}, 0).to(torch::kFloat32); //[2, 47, 156]
    coords_0 = coords_0.unsqueeze(0).repeat({batch, 1, 1, 1}); 
    Tensor coords_1 = coords_0.clone();
    return {coords_0, coords_1};
}

tuple<Tensor, Tensor> calcflow_raft::fnet_forward(Tensor& tensor0, Tensor& tensor1) {
    void *buffer[4]{};
    auto sizes = tensor0.sizes();
    int index0 = fnet_engine->getBindingIndex("img0");
    int index1 = fnet_engine->getBindingIndex("img1");
    buffer[index0] = tensor0.data_ptr();
    buffer[index1] = tensor1.data_ptr();
    //动态时，也需要设置输入维度
    nvinfer1::Dims4 input_dim1;
    input_dim1.d[0] = sizes[0];
    input_dim1.d[1] = sizes[1];
    input_dim1.d[2] = sizes[2];
    input_dim1.d[3] = sizes[3];
    fnet_context->setBindingDimensions(index0, input_dim1);
    fnet_context->setBindingDimensions(index1, input_dim1);

    auto opt = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);

    int index2 = fnet_engine->getBindingIndex("fmap1");
    int index3 = fnet_engine->getBindingIndex("fmap2");
    Tensor fmat0 = torch::zeros({1, 128, sizes[2] / 8, sizes[3] / 8}, opt); //设置输出维度
    Tensor fmat1 = torch::zeros({1, 128, sizes[2] / 8, sizes[3] / 8}, opt); //设置输出维度
    buffer[index2] = fmat0.data_ptr(); //index0 = 2
    buffer[index3] = fmat1.data_ptr(); //index1 =3

    fnet_context->enqueueV2(buffer, stream, nullptr); //这个应该就是核心
    // return {fmat0.to(torch::kFloat), fmat1.to(torch::kFloat)};
    return {fmat0, fmat1};
}

Tensor calcflow_raft::cnet_forward(Tensor& tensor0) {
    void *buffer[2]{};
    auto sizes = tensor0.sizes();
    int index0 = cnet_engine->getBindingIndex("img0");
    buffer[index0] = tensor0.data_ptr();
    //动态时，也需要设置输入维度
    nvinfer1::Dims4 input_dim1;
    input_dim1.d[0] = sizes[0];
    input_dim1.d[1] = sizes[1];
    input_dim1.d[2] = sizes[2];
    input_dim1.d[3] = sizes[3];
    cnet_context->setBindingDimensions(index0, input_dim1);

    auto opt = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);

    int index1 = cnet_engine->getBindingIndex("fmap1");
    Tensor fmat0 = torch::zeros({1, 160, sizes[2] / 8, sizes[3] / 8}, opt); //设置输出维度
    buffer[index1] = fmat0.data_ptr(); //index0 = 2

    cnet_context->enqueueV2(buffer, stream, nullptr); //这个应该就是核心
    // cout << "fmat0.sizes()" << fmat0.sizes() << endl;
    return fmat0.to(torch::kFloat);
    return fmat0;
}

tuple<Tensor, Tensor, Tensor> calcflow_raft::decoder_forward(Tensor& tensor0, Tensor& tensor1, Tensor& tensor2, Tensor& tensor3) {
    void *buffer[7]{};
    auto sizes_0 = tensor0.sizes();
    auto sizes_1 = tensor1.sizes();
    auto sizes_2 = tensor2.sizes();
    auto sizes_3 = tensor3.sizes();
    int index0 = decoder_engine->getBindingIndex("net");
    int index1 = decoder_engine->getBindingIndex("inp");
    int index2 = decoder_engine->getBindingIndex("corr");
    int index3 = decoder_engine->getBindingIndex("flow");
    buffer[index0] = tensor0.data_ptr();
    buffer[index1] = tensor1.data_ptr();
    buffer[index2] = tensor2.data_ptr();
    buffer[index3] = tensor3.data_ptr();
    //动态时，也需要设置输入维度
    nvinfer1::Dims4 input_dim0;
    input_dim0.d[0] = sizes_0[0];
    input_dim0.d[1] = sizes_0[1];
    input_dim0.d[2] = sizes_0[2];
    input_dim0.d[3] = sizes_0[3];

    nvinfer1::Dims4 input_dim1;
    input_dim1.d[0] = sizes_1[0];
    input_dim1.d[1] = sizes_1[1];
    input_dim1.d[2] = sizes_1[2];
    input_dim1.d[3] = sizes_1[3];

    nvinfer1::Dims4 input_dim2;
    input_dim2.d[0] = sizes_2[0];
    input_dim2.d[1] = sizes_2[1];
    input_dim2.d[2] = sizes_2[2];
    input_dim2.d[3] = sizes_2[3];

    nvinfer1::Dims4 input_dim3;
    input_dim3.d[0] = sizes_3[0];
    input_dim3.d[1] = sizes_3[1];
    input_dim3.d[2] = sizes_3[2];
    input_dim3.d[3] = sizes_3[3];

    decoder_context->setBindingDimensions(index0, input_dim0);
    decoder_context->setBindingDimensions(index1, input_dim1);
    decoder_context->setBindingDimensions(index2, input_dim2);
    decoder_context->setBindingDimensions(index3, input_dim3);

    auto opt = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);

    int index4 = decoder_engine->getBindingIndex("net_out");
    int index5 = decoder_engine->getBindingIndex("mask");
    int index6 = decoder_engine->getBindingIndex("delta_flow");
    Tensor fmat0 = torch::zeros({1, 96, sizes_0[2], sizes_0[3]}, opt); 
    Tensor fmat1 = torch::zeros({1, 576, sizes_0[2], sizes_0[3]}, opt); 
    Tensor fmat2 = torch::zeros({1, 2, sizes_0[2], sizes_0[3]}, opt); 
    buffer[index4] = fmat0.data_ptr(); 
    buffer[index5] = fmat1.data_ptr(); 
    buffer[index6] = fmat2.data_ptr(); 

    decoder_context->enqueueV2(buffer, stream, nullptr); //核心
    return {fmat0, fmat1, fmat2};
}

Tensor calcflow_raft::trace_predict(Tensor& img1_tensor, Tensor& img2_tensor) {
    auto fnet_result = fnet_forward(img1_tensor, img2_tensor);
    Tensor fmap1 = get<0>(fnet_result);
    Tensor fmap2 = get<1>(fnet_result);

    Corr(fmap1, fmap2);

    Tensor cmap = cnet_forward(img1_tensor);

    // trace_time.tic();
    // auto split_cmap = torch::split_with_sizes(cmap, {128, 128}, 1);
    auto split_cmap = torch::split_with_sizes(cmap, {96, 64}, 1);
    Tensor net = torch::tanh(split_cmap[0]);//[1,128,47,154]
    Tensor inp = torch::relu(split_cmap[1]);
    //初始化光流
    auto result = Initialize(fmap1);
    Tensor coords_0 = std::get<0>(result); //[1, 2, 47, 156]
    Tensor coords_1 = std::get<1>(result);
    // cout << "pre_spend_time: " << trace_time.toc() << endl;
    // trace_time.tic();
    int iters = 5;
    Tensor flow_predictions;
    for (int i = 0; i < iters; i++) {
        coords_1 = coords_1.detach();
        //查找相似度
        Tensor corr = Index_correlation(coords_1, true);
        Tensor flow = coords_1 - coords_0;

        auto update_result = decoder_forward(net, inp, corr, flow);
        net = get<0>(update_result);
        Tensor mask = get<1>(update_result);
        Tensor delta_flow = get<2>(update_result);

        coords_1 += delta_flow;
        if (i == iters -1)
            flow_predictions = upsample_flow(coords_1 - coords_0, mask);
    }
    return flow_predictions;
}

void calcflow_raft::raft_predict(cv::Mat &img1, cv::Mat &img2, vector<cv::Point2f>&kps_last, vector<cv::Point2f>&kps_curr, vector<uchar>&status) {
    // std::cout << "img1.channels()" << img1.channels() << "img1.size()" << img1.size() << std::endl;
    cout << "---------------raft version------------------" << endl;
    // sys_time.tic();
    // trace_time.tic();
    Tensor img1_tensor = convert->process(img1);
    auto img1_tensor_size = img1_tensor.sizes();
    int high = img1_tensor_size[2];
    int wide = img1_tensor_size[3];
    Tensor img2_tensor = convert->process(img2);
    // cout << "convert.toc()" << trace_time.toc() << endl;
    // auto flow = raft_pred->Forward(img1_tensor, img2_tensor);
    trace_time.tic();
    auto flow = trace_predict(img1_tensor, img2_tensor);
    // cout << "trace_predict.toc()" << trace_time.toc() << endl;

    // trace_time.tic();
    flow = flow.squeeze(); //[2, 440, 1024]
//--------------------------显示----------------------------
    auto flow_final_show = flow.clone();
    cv::Mat flo_img = visual_flow_image(flow_final_show);
    cv::imshow("flowssssssss", flo_img);
    cv::waitKey(1);
//---------------------------------------------------------  
    trace_time.tic();
    Tensor flow_index = flow.permute({1, 2, 0}).to(torch::kCPU); //[440, 1024, 2]
    auto flow_index_access = flow_index.accessor<float, 3>();

    cv::Point2f point_2;
    for (auto &kp : kps_last) {
        int y_min = static_cast<int>(kp.y);
        int y_max = ((y_min + 1) > high ) ? high : (y_min + 1);
        int x_min = static_cast<int>(kp.x);
        int x_max = ((x_min + 1) > wide ) ? wide : (x_min + 1);
        // cout << "kp.x: " << kp.x << "  kp.y: " << kp.y << "  x_min: " << x_min << "  x_max: " 
        // << x_max << "  y_min: " << y_min << "  y_max: " << y_max << endl;

        //先对x作两次线性插值
        float dx_min_1 = flow_index_access[y_min][x_min][0];
        float dx_max_1 = flow_index_access[y_min][x_max][0];
        float dx_1 = dx_min_1 + (dx_max_1 - dx_min_1) * (kp.x - x_min);
        float dy_min_1 = flow_index_access[y_min][x_min][1];
        float dy_max_1 = flow_index_access[y_min][x_max][1];
        float dy_1 = dy_min_1 + (dy_max_1 - dy_min_1) * (kp.x - x_min);
        
        float dx_min_2 = flow_index_access[y_max][x_min][0];
        float dx_max_2 = flow_index_access[y_max][x_max][0];
        float dx_2 = dx_min_2 + (dx_max_2 - dx_min_2) * (kp.x - x_min);
        float dy_min_2 = flow_index_access[y_max][x_min][1];
        float dy_max_2 = flow_index_access[y_max][x_max][1];
        float dy_2 = dy_min_2 + (dy_max_2 - dy_min_2) * (kp.x - x_min);
        
        float dx = dx_1 + (dx_2 - dx_1) * (kp.y - y_min);
        float dy = dy_1 + (dy_2 - dy_1) * (kp.y - y_min);

        point_2.x = kp.x + dx;
        point_2.y = kp.y + dy;
        // cout << "point_2.x" << point_2.x << endl;
        kps_curr.push_back(point_2);
        double motion = sqrt(dx * dx + dy * dy);

        if((point_2.x > wide) || (point_2.y > high) || (point_2.x < 0) || (point_2.y < 0) || (std::isnan(point_2.x)) || (std::isnan(point_2.y))) {
            status.push_back(0);
        } else {
            status.push_back(1);
        }
    }
    // cout << "point_2.toc()" << trace_time.toc() << endl;
    // cout << "sys.toc()" << sys_time.toc() << endl;
}

