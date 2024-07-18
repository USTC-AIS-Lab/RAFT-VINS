#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <string>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>
#include <opencv2/features2d.hpp>
#include "Raft_torch.h"
#include "Pipeline.h"
#include "Visual.h"
#include "utils.h"
#include "calcflow_raft.h"

#define OPENCV_TRAITS_ENABLE_DEPRECATED

using namespace std;
using namespace cv;

string file_1 = "../paper/two-floors-csc1/1.png";  // first image
string file_2 = "../paper/two-floors-csc1/2.png";  // second image

class Dataset{
  public:
    Dataset(const char* image_path):image_path_(image_path){}
    cv::Mat ReadImage(int index);

    const char* image_path_;
};

cv::Mat Dataset::ReadImage(int index){
    boost::format fmt("%s/%06d.png");
    string image_file = (fmt % image_path_ % index).str();
    // boost::format fmt("%s/%s%04d.png");
    // string image_file = (fmt % image_path_ % "frame_" % index).str();
    // cout << "image_file: " << image_file << endl;
    cv::Mat image, image_resize;
    image = cv::imread(image_file, cv::IMREAD_GRAYSCALE); //cv::Mat img1 =  cv::imread(file1, CV_LOAD_IMAGE_COLOR); cv::IMREAD_GRAYSCALE
    if (image.empty()) {
        cerr << "Failed to read image: " << image_file << endl;
    }
    // image.resize()
    
    cv::resize(image, image_resize, cv::Size(), 0.5, 0.5,
               cv::INTER_NEAREST);
    return image_resize;
    // return image;
}

int main(int argc, char** argv) {
    string filename_1 = argv[1];
    string filename_2 = argv[2];
    string method = argv[3];
    int method_int = stoi(method);
    // cout << "method_int: " << method_int << endl;
    // exit(0);
    cv::Mat img0 = cv::imread(filename_1, 0);
    cv::Mat img1 = cv::imread(filename_2, 0);
    vector<cv::KeyPoint> keypoints1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(150, 0.01, 20); // maximum 500 keypoints
    detector->detect(img0, keypoints1);
    std::unique_ptr<calcflow_raft> calc_flow = std::make_unique<calcflow_raft>();
    std::unique_ptr<Pipeline> convert = std::make_unique<Pipeline>();
    Tensor img0_tensor = convert->process(img0);
    Tensor img1_tensor = convert->process(img1);
    vector<cv::Point2f> kps_last, kps_curr;
    for(auto kp1 : keypoints1) {
        kps_last.push_back(kp1.pt);
    }
    std::vector<uchar>status;
    vector<float> err;
    cout << "------------------------------------------------------" << endl;
    if (method_int)
        calc_flow->raft_predict(img0, img1, kps_last, kps_curr, status);//还需要返回status，追踪到为1否则为0
    else
        cv::calcOpticalFlowPyrLK(img0, img1, kps_last, kps_curr, status, err, cv::Size(21, 21), 3);
    vector<KeyPoint> kps2(int(keypoints1.size()));
    for (int i = 0; i < keypoints1.size(); i++) {
        kps2[i].pt = kps_curr[i];
    }
    vector<cv::DMatch> matchs;
    for (int i = 0; i < keypoints1.size(); i++) {
        if (status[i]) {
            cv::DMatch match;
            match.queryIdx = i;
            match.trainIdx = i;
            matchs.push_back(match);
        }
    }
    // 垂直拼接两张图片
    Mat img0_color, img1_color;
    Mat combined_img, combined_img2;
    cv::cvtColor(img0, img0_color, COLOR_GRAY2RGB);
    cv::cvtColor(img1, img1_color, COLOR_GRAY2RGB);
    vconcat(img0_color, img1_color, combined_img); //DRAW_OVER_OUTIMG
    for (auto nkp1 : keypoints1) {
        cv::circle(combined_img, nkp1.pt, 5, cv::Scalar(255, 0, 0), -1);
    }
    for (int i = 0; i < kps2.size(); i++) {
        // nkp2.pt.y += img0_color.rows;
        cv::KeyPoint kps = kps2[i];
        kps.pt.y += img0_color.rows;
        // cout << "img0_color.rows: " << img0_color.rows << endl;
        if(kps.pt.y > img0_color.rows) {
            cv::circle(combined_img, kps.pt, 5, cv::Scalar(0, 0, 255), -1);
        }
    }
    for (int i = 0; i < keypoints1.size(); i++) {
        kps2[i].pt.y += img0_color.rows;
        // if (fabs((kps2[i].pt.y - img0_color.rows - keypoints1[i].pt.y) + (kps2[i].pt.x - keypoints1[i].pt.x)) < 50) {
            if(1){
                line(combined_img, keypoints1[i].pt, kps2[i].pt, cv::Scalar(0, 255, 0));
                // cout << "distance: " << ((kps2[i].pt.y - img0_color.rows - keypoints1[i].pt.y) + (kps2[i].pt.x - keypoints1[i].pt.x)) << endl;
            }
        else
            line(combined_img, keypoints1[i].pt, kps2[i].pt, cv::Scalar(204, 0, 204));
    }


    // cv::Mat img_match;
    // drawMatches(img0, keypoints1, img1, kps2, matchs, img_match);
    // cv::imshow("tracked by opencv", img_match);
    // drawMatches(img0, keypoints1, img1, kps2, matchs, combined_img, 
                    // Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::DRAW_OVER_OUTIMG);
    // cv::imshow("tracked by opencv", combined_img);
    // cv::cvtColor(img0, img0, COLOR_GRAY2RGB);
    cv::imshow("combined_img", combined_img);
    cv::imwrite("20240202-0.jpg", combined_img);
    cv::waitKey(0);

    return 0;
}

// int main(int argc, char** argv) {
//     std::unique_ptr<Dataset> dataset = std::make_unique<Dataset>("/home/zj/datasets/KITTI/data_odometry_gray/dataset/sequences/00/image_0");
//     std::unique_ptr<calcflow_raft> calcflow = std::make_unique<calcflow_raft>();
//     cv::Ptr<cv::GFTTDetector> gftt = cv::GFTTDetector::create(150, 0.01, 20); 
//     cv::Mat img0 = dataset->ReadImage(0);
//     vector<cv::KeyPoint> keypoints1;
//     gftt->detect(img0, keypoints1);
//     cv::Mat out_img0;
//     cv::drawKeypoints(img0, keypoints1, out_img0, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
//     // cv::imshow("gftt", out_img0);
//     // cv::waitKey(1);
//     for (int i = 1; i < 10000; i++) {
//         Tictoc tt;
//         cv::Mat img1 = dataset->ReadImage(i);
//         if (keypoints1.size() < 30) {
//             gftt->detect(img0, keypoints1);
//         }
//         // cout <<  "keypoints1.size()" << keypoints1.size() << endl;

//         vector<cv::Point2f> kps_last, kps_curr;
//         for(auto kp1 : keypoints1) {
//             kps_last.push_back(kp1.pt);
//         }
//         std::vector<uchar>status;
//         cout << "------------------------------------------------------" << endl;
//         tt.tic();
//         calcflow->raft_predict(img0, img1, kps_last, kps_curr, status);//还需要返回status，追踪到为1否则为0
//         std::cout << "spend time: " << tt.toc() << endl; //spend time: 69.6683
//         // cout <<  "kps_curr.size()" << kps_curr.size() << endl;

//         int num_good_tracking = 0;
//         vector<cv::KeyPoint> keypoints2;
//         for (int j = 0; j < status.size(); j++) {
//             if (status[j]) {
//                 cv::KeyPoint new_kp(kps_curr[j], 7);
//                 keypoints2.push_back(new_kp);
//                 num_good_tracking++;
//             }
//         }
//         // cout << "num_good_tracking: " << num_good_tracking << endl;
//         cv::Mat out_img1;
//         cv::drawKeypoints(img1, keypoints2, out_img1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
//         cv::imshow("afterflow", out_img1);
//         cv::waitKey(1);
//         img0 = img1;
//         keypoints1 = keypoints2;
//     }
// }
  
