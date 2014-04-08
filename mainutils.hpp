#pragma once

#include <functional>
#include <string>

namespace cv {
    class Mat;
}

namespace Json {
    class Value;
}

namespace warco {

    void foreach_img(const Json::Value& dataset, const char* traintest,
                     std::function<void (unsigned, const cv::Mat&, std::string)> fn);

} // namespace warco

