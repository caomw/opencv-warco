#pragma once

#include <functional>

namespace cv {
    class Mat;
}

namespace warco {

    cv::Mat eig_fn(const cv::Mat& m, std::function<double (double)> fn);
    cv::Mat mkspd(const cv::Mat& m);
    cv::Mat randspd(unsigned rows, unsigned cols);
    void assert_mat_almost_eq(const cv::Mat& actual, const cv::Mat& expected);

} // namespace warco

