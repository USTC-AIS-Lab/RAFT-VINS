#include "utils.h"

using namespace std;

Tensor pad_tensor(Tensor& image1) {
    // cout << "image1.sizes(): " << image1.sizes() << endl;
    int high = image1.sizes()[2];
    int wide = image1.sizes()[3];
    // cout << "wide: " << wide << "hight: " << high << endl;
    int pad_wd = ((((wide / 8) + 1 ) * 8 - wide) % 8);
    int pad_ht = ((((high / 8) + 1 ) * 8 - high) % 8);
    // cout << "pad_wd: " << pad_wd << "pad_ht: " << pad_ht << endl;
    torch::nn::functional::PadFuncOptions pad_tensor({(pad_wd / 2), pad_wd - (pad_wd / 2), (pad_ht / 2), pad_ht - (pad_ht / 2)});
    auto image1_tensor = torch::nn::functional::pad(image1, pad_tensor.mode(torch::kReplicate));
    // auto image1_tensor = torch::nn::functional::pad(image2, pad_tensor.mode(torch::kReplicate));
    return image1_tensor;
}



