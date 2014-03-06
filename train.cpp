#include <fstream>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include "json/json.h"

#include "covcorr.hpp"
#include "dists.hpp"
#include "features.hpp"
//#include "model.hpp"

int main(int argc, char** argv)
{
    if(argc != 2) {
        std::cout << "Please specify a configuration file which describes the dataset." << std::endl;
        return 0;
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
    std::string fname = dataset["train"]["front"][0].asString();
    cv::Mat image = cv::imread(fname);
    if(!image.data) {
        std::cerr << "Failed to load test image " << fname << std::endl;
        return 4;
    }
    imshow("Foo", image);
    cv::waitKey(0);

    auto feats = warco::mkfeats(image);
    warco::showfeats(feats);

    std::cout << "corr " << warco::extract_corrs(feats)[0] << std::endl;

    //corrs = [corr for ...]
    //labels = ...;
    //auto model0(corrs, labels)

    return 0;
}
