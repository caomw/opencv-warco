#include <fstream>
#include <iostream>
#include <string>

#include "json/json.h"

#include "filterbank.hpp"
#include "mainutils.hpp"
#include "warco.hpp"

int main(int argc, char** argv)
{
    if(argc != 2) {
        std::cout << "Please specify a configuration file which describes the dataset." << std::endl;
        return 0;
    }

    Json::Value dataset;
    {
        std::ifstream infile(argv[1]);
        if(! infile) {
            std::cerr << "Failed to open configuration file " << argv[1] << std::endl;
            return 2;
        }

        Json::Reader reader;
        if(! reader.parse(infile, dataset)) {
            std::cerr << "Failed to parse configuration file " << argv[1] << std::endl
                << reader.getFormattedErrorMessages()
                << std::endl;
            return 3;
        }
    }

    // Read in the patch definitions from the config file
    // or default if none is defined.
    std::vector<warco::Patch> patches;
    if(dataset.isMember("patches")) {
        for(const Json::Value& p : dataset["patches"]) {
            if(p.size() != 4)
                throw std::runtime_error("One of the patches specified in " + std::string(argv[1]) + " is uncorrectly specified as it doesn't have four entries.");

            patches.push_back(warco::Patch{p[0].asDouble(), p[1].asDouble(), p[2].asDouble(), p[3].asDouble()});
        }
    } else {
        // Default from warco for 50x50
        for(auto y = 0 ; y < 5 ; ++y)
            for(auto x = 0 ; x < 5 ; ++x)
                patches.push_back(warco::Patch{(1+8*x)/50., (1+8*y)/50., 16/50., 16/50.});
    }

    auto fb = cv::FilterBank(dataset["filterbank"].asCString());
    warco::Warco model(fb, patches);
    warco::foreach_img(dataset, "train", [&model](unsigned lbl, const cv::Mat& image, std::string) {
        model.add_sample(image, lbl);
    });

    std::cout << "Training model" << std::flush;
    std::vector<double> C;
    if(dataset.isMember("crossval_C"))
        for(Json::Value c : dataset["crossval_C"])
            C.push_back(c.asDouble());
    else
        C = {0.1, 1.0, 10.0};
    double avg_train = model.train(C, [](unsigned){ std::cout << "." << std::flush; });
    std::cout << std::endl << "Average training score *per patch*: " << avg_train << std::endl;

    std::cout << "Testing" << std::flush;
    std::cerr << "test,predicted,actual" << std::endl;

    Json::Value lbls = dataset["classes"];
    unsigned correct = 0, total = 0;
    warco::foreach_img(dataset, "test", [&model, &lbls, &correct, &total](unsigned lbl, const cv::Mat& image, std::string fname) {
        std::cout << "." << std::flush;

        //unsigned pred = model.predict(image);
        unsigned pred = model.predict_proba(image);

        std::cerr << fname << "," << lbls[pred].asString() << "," << lbls[lbl].asString() << std::endl;

        correct += pred == lbl;
        ++total;
    });

    std::cout << std::endl << "score: " << 100.0*correct/total << "%" << std::endl;

    return 0;
}
