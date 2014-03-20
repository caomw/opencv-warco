#include <fstream>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include "json/json.h"

#include "covcorr.hpp"
#include "dists.hpp"
#include "features.hpp"
#include "model.hpp"
#include "cvutils.hpp"

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

#ifndef NDEBUG
#include <iomanip>
template<typename T>
std::string to_s(const std::vector<T>& v)
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3) << "[";

    auto Nm1 = v.size() - 1;
    for(unsigned i = 0 ; i < Nm1 ; ++i) {
        ss << v[i] << ", ";
    }
    ss << v[Nm1] << "]";

    return ss.str();
}
#endif // NDEBUG

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

    // TODO: refactor the models stuff into one class and
    //       make the magic numbers configurable.
    std::vector<warco::PatchModel> models(5*5);
    foreach_img(dataset, "train", [&models](unsigned lbl, const cv::Mat& image, std::string) {
        auto corrs = warco::extract_corrs(warco::mkfeats(image));
        auto icorr = corrs.begin();
        auto imodel = models.begin();
        for( ; icorr != corrs.end() && imodel != models.end() ; ++icorr, ++imodel ) {
            imodel->add_sample(*icorr, lbl);
        }
    });

    std::cout << "Training models" << std::flush;
    for(auto& model : models) {
        std::cout << "." << std::flush;
        model.train();
    }
    std::cout << std::endl;

    std::cout << "Testing" << std::flush;
    std::cerr << "test,predicted,actual" << std::endl;

    Json::Value lbls = dataset["classes"];
    unsigned L = lbls.size();
    unsigned correct = 0, total = 0;
    foreach_img(dataset, "test", [&](unsigned lbl, const cv::Mat& image, std::string fname) {
        std::cout << "." << std::flush;

        auto corrs = warco::extract_corrs(warco::mkfeats(image));
        auto icorr = begin(corrs);
        auto imodel = begin(models);
        auto ecorr = end(corrs);
        auto emodel = end(models);

        std::vector<unsigned> votes(L, 0);
        std::vector<double> votes_proba(L, 0.0);
        for( ; icorr != ecorr && imodel != emodel ; ++icorr, ++imodel ) {
            unsigned pred = imodel->predict(*icorr);
            votes[pred] += 1;
#ifndef NDEBUG
            if(vv)
                std::cout << " " << pred;
#endif

            std::vector<double> probas = imodel->predict_probas(*icorr);
            double w = 1.0 / models.size();
            for(unsigned i = 0 ; i < L ; ++i) {
                votes_proba[i] += probas[i] * w;
            }
        }

        // argmax
        unsigned pred = std::max_element(begin(votes), end(votes)) - begin(votes);
        unsigned pred_proba = std::max_element(begin(votes_proba), end(votes_proba)) - begin(votes_proba);

#ifndef NDEBUG
        if(vv) {
            std::cout << " => " << pred << " (=" << lbls[pred].asString() << ")"<< std::endl;
            std::cout << to_s(votes) << std::endl;
            std::cout << to_s(votes_proba) << std::endl;
            std::cout << " => " << pred_proba << " (=" << lbls[pred_proba].asString() << ")"<< std::endl;
        }
#endif

        std::cerr << fname << "," << lbls[pred].asString() << "," << lbls[lbl].asString() << std::endl;
        //std::cerr << fname << "," << lbls[pred_proba].asString() << "," << lbls[lbl].asString() << std::endl;

        correct += pred == lbl;
        ++total;
    });

    std::cout << std::endl << "score: " << 100.0*correct/total << "%" << std::endl;

    return 0;
}
