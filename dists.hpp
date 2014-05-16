#pragma once

#include <functional>
#include <string>
#include <vector>

namespace cv {
    class Mat;
}

namespace warco {

    typedef std::function<float(const cv::Mat&, const cv::Mat&)> distfn_t;

    distfn_t get_distfn(const std::string& name);
    std::string get_name(distfn_t);

    void test_dists();
    float dist_euc(const cv::Mat& corrA, const cv::Mat& corrB);
    float dist_cbh(const cv::Mat& corrA, const cv::Mat& corrB);
    float dist_geo(const cv::Mat& corrA, const cv::Mat& corrB);

    float dist_my_euc(const cv::Mat& corrA, const cv::Mat& corrB);
} // namespace warco

