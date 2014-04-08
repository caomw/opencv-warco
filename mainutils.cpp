#include "mainutils.hpp"

#include <opencv2/opencv.hpp>
#include "json/json.h"

void warco::foreach_img(const Json::Value& dataset, const char* traintest, std::function<void (unsigned, const cv::Mat&, std::string)> fn)
{
    auto lbls = dataset["classes"];
    for(auto ilbl = lbls.begin() ; ilbl != lbls.end() ; ++ilbl) {
        auto lbl = ilbl.index();
        auto lblname = (*ilbl).asString();

        for(Json::Value fname : dataset[traintest][lblname]) {
            cv::Mat image = cv::imread(fname.asString());
            if(! image.data) {
                std::cerr << "Skipping unreadable image " << fname << std::endl;
                continue;
            }

            fn(lbl, image, fname.asString());
        }
    }
}

