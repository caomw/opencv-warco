#include <fstream>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include "json/json.h"

#include "covcorr.hpp"
#include "cvutils.hpp"
#include "dists.hpp"
#include "features.hpp"
#include "model.hpp"
#include "warco.hpp"

void foreach_img(const Json::Value& dataset, const char* traintest, std::function<void (unsigned, const cv::Mat&, std::string)> fn)
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

int main(int argc, char** argv)
{
    if(argc != 2) {
        std::cout << "Please specify a configuration file which describes the dataset." << std::endl;
        return 0;
    }

#ifndef NDEBUG
    bool vv = getenv("WARCO_DEBUG");
#endif

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

    warco::Warco model;
    foreach_img(dataset, "train", [&model](unsigned lbl, const cv::Mat& image, std::string) {
        model.add_sample(image, lbl);
    });

    std::cout << "Training models" << std::flush;
    double avg_train = model.train();
    std::cout << std::endl << "Average training score: " << avg_train << std::endl;

    std::cout << "Testing" << std::flush;
    std::cerr << "test,predicted,actual" << std::endl;

    Json::Value lbls = dataset["classes"];
    unsigned correct = 0, total = 0;
    foreach_img(dataset, "test", [&model, &lbls, &correct, &total](unsigned lbl, const cv::Mat& image, std::string fname) {
        std::cout << "." << std::flush;

        unsigned pred = model.predict(image);
        //unsigned pred_proba = model.predict_proba(image);

#ifndef NDEBUG
        //if(vv) {
            //std::cout << " => " << pred << " (=" << lbls[pred].asString() << ")"<< std::endl;
            //std::cout << to_s(votes) << std::endl;
            //std::cout << to_s(votes_proba) << std::endl;
            //std::cout << " => " << pred_proba << " (=" << lbls[pred_proba].asString() << ")"<< std::endl;
        //}
#endif

        std::cerr << fname << "," << lbls[pred].asString() << "," << lbls[lbl].asString() << std::endl;
        //std::cerr << fname << "," << lbls[pred_proba].asString() << "," << lbls[lbl].asString() << std::endl;

        correct += pred == lbl;
        ++total;
    });

    std::cout << std::endl << "score: " << 100.0*correct/total << "%" << std::endl;

    return 0;
}
