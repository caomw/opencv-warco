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

    Json::Value dataset = warco::readJson(argv[1]);
    auto patches = warco::readPatches(dataset);
    auto fb = cv::FilterBank(dataset["filterbank"].asCString());
    auto dfn = dataset.get("dist", "cbh").asString();
    bool cov = dataset.get("cov", "true").asBool();
    warco::Warco model(fb, patches, dfn);

    std::cout << "Loading images... " << std::flush;
    warco::foreach_img(dataset, "train", [&model](unsigned lbl, const cv::Mat& image, std::string) {
        model.add_sample(image, lbl);
    });
    std::cout << "Done" << std::endl;

    std::cout << "Training model with:" << std::endl
        << "- filterbank: " << dataset["filterbank"].asString() << std::endl
        << "- distance: " << dfn << std::endl
        << "- #patches: " << patches.size() << std::endl
        << "- " << (cov ? "covariances" : "correlations") << std::endl;
    auto C = warco::readCrossvalCs(dataset);

    if(!cov)
        model.prepare();

    double avg_train = model.train(C, [](){ std::cout << "." << std::flush; });
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

