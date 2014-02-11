#pragma once

#include <functional>

namespace cv {
    class Mat;
}

namespace warco {

    cv::Mat eig_fn(const cv::Mat& m, std::function<double (double)> fn);
    void assert_mat_almost_eq(const cv::Mat& actual, const cv::Mat& expected);

} // namespace warco

