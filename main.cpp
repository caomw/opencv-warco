#include <fstream>
#include <iostream>

#include <opencv2/opencv.hpp>
#include "json/json.h"

#include "features.hpp"

int main(int argc, char** argv)
{
    if(argc != 2) {
        std::cerr << "Giff config file as argument!" << std::endl;
        return 1;
    }

    Json::Value dataset;
    {
        std::ifstream infile(argv[1]);
        if(!infile) {
            std::cerr << "Failed to open configuration file " << argv[1] << std::endl;
            return 2;
        }

        Json::Reader reader;
        if(!reader.parse(infile, dataset)) {
            std::cerr << "Failed to parse configuration file " << argv[1] << std::endl
                << reader.getFormattedErrorMessages()
                << std::endl;
            return 3;
        }
    }

    // Compute the features of all train images.
    cv::Mat image = cv::imread(dataset["train"]["front"][0].asString());
    imshow("Foo", image);
    cv::waitKey(0);

    auto feats = warco::mkfeats(image);
    warco::showfeats(feats);

    return 0;
}
