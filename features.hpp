#pragma once

#include <vector>

namespace cv {
    class Mat;
}

namespace warco {

    using Features = std::vector<cv::Mat>;

    Features mkfeats(const cv::Mat& m);
    void showfeats(const Features& feats);

} // namespace warco

