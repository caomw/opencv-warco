#include "warco.hpp"

#include <unordered_map>

#include <opencv2/imgproc.hpp>

#include "covcorr.hpp"
#include "features.hpp"
#include "model.hpp"
#include <iostream>

#ifndef NDEBUG
#include <iomanip>
#include <iostream>
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

warco::Warco::Patch::Patch()
    : w(0.0)
    , model(new PatchModel())
{ }

warco::Warco::Warco()
    : _patchmodels(5*5)
{ }

warco::Warco::~Warco()
{ }

void warco::Warco::add_sample(const cv::Mat& img, unsigned label)
{
    this->foreach_model(img, [label](const Patch& patch, const cv::Mat& corr) {
        patch.model->add_sample(corr, label);
    });
}

double warco::Warco::train()
{
    for(auto& model : _patchmodels) {
#ifndef NDEBUG
        if(getenv("WARCO_DEBUG")) {
            std::cout << "." << std::flush;
        }
#endif
        model.model->train();
        model.w = 1.0/_patchmodels.size();
    }

    // TODO: Actually would be nice to return training error if it's free.
    return 0.0;
}

unsigned warco::Warco::predict(const cv::Mat& img) const
{
    std::vector<double> votes(_patchmodels.front().model->nlbls(), 0.0);

    this->foreach_model(img, [&votes](const Patch& patch, const cv::Mat& corr) {
        unsigned pred = patch.model->predict(corr);
        votes[pred] += patch.w;

#ifndef NDEBUG
        if(getenv("WARCO_DEBUG")) {
            std::cout << " " << pred;
        }
#endif
    });

#ifndef NDEBUG
    if(getenv("WARCO_DEBUG")) {
        std::cout << to_s(votes) << std::endl;
    }
#endif

    // argmax
    return std::max_element(begin(votes), end(votes)) - begin(votes);
}

unsigned warco::Warco::predict_proba(const cv::Mat& img) const
{
    std::vector<double> probas(_patchmodels.front().model->nlbls(), 0.0);

    this->foreach_model(img, [&probas](const Patch& patch, const cv::Mat& corr) {
        auto pred = patch.model->predict_probas(corr);

        for(unsigned i = 0 ; i < probas.size() ; ++i)
            probas[i] += pred[i] * patch.w;

#ifndef NDEBUG
        if(getenv("WARCO_DEBUG")) {
            std::cout << " " << to_s(pred);
        }
#endif
    });

#ifndef NDEBUG
    if(getenv("WARCO_DEBUG")) {
        std::cout << to_s(probas) << std::endl;
    }
#endif

    // argmax
    return std::max_element(begin(probas), end(probas)) - begin(probas);
}

void warco::Warco::foreach_model(const cv::Mat& img, std::function<void(const Patch& patch, const cv::Mat& corr)> fn) const
{
    cv::Mat img50(50, 50, img.type());

    // Resize if necessary. Could also be smart and create grid with
    // relative sizes etc. but meh. TODO
    if(img.cols != 50 || img.rows != 50) {
        cv::resize(img, img50, img50.size());
    } else {
        img50 = img;
    }

    auto feats = warco::mkfeats(img50);

    for(auto y = 0 ; y < 5 ; ++y)
        for(auto x = 0 ; x < 5 ; ++x)
            fn(_patchmodels[y*5 + x], extract_corr(feats, 1+8*x, 1+8*y, 16, 16));
}

