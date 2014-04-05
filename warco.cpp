#include "warco.hpp"

#include <opencv2/imgproc.hpp>

#include "covcorr.hpp"
#include "features.hpp"
#include "model.hpp"

#ifndef NDEBUG
#  include <iostream>
#  include "to_s.hpp"
#endif

warco::Warco::Patch::Patch()
    : w(0.0)
    , model(new PatchModel())
{ }

warco::Warco::Warco(cv::FilterBank fb)
    : _patchmodels(5*5)
    , _fb(fb)
{ }

warco::Warco::~Warco()
{ }

void warco::Warco::add_sample(const cv::Mat& img, unsigned label)
{
    this->foreach_model(img, [label](const Patch& patch, const cv::Mat& corr) {
        patch.model->add_sample(corr, label);
    });
}

double warco::Warco::train(const std::vector<double>& cvC, std::function<void(unsigned)> progress)
{
    double w_tot = 0.0;
    for(auto& patch : _patchmodels) {
        patch.w = patch.model->train(cvC);
        w_tot += patch.w;

        progress(_patchmodels.size() + 1);
    }

    for(auto& patch : _patchmodels) {
        patch.w /= w_tot;
    }

    progress(_patchmodels.size() + 1);

#ifndef NDEBUG
    if(getenv("WARCO_DEBUG")) {
        unsigned x = 0, w = static_cast<unsigned>(sqrt(_patchmodels.size()));
        for(const auto& patch : _patchmodels) {
            if(x++ % w == 0)
                std::cout << std::endl;
            std::cout << patch.w << " ";
        }
    }
#endif

    // Return the average error.
    return w_tot / _patchmodels.size();
}

unsigned warco::Warco::predict(const cv::Mat& img) const
{
    std::vector<double> votes(this->nlbl(), 0.0);

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
    std::vector<double> probas(this->nlbl(), 0.0);

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

unsigned warco::Warco::nlbl() const
{
    return _patchmodels.front().model->nlbls();
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

    auto feats = warco::mkfeats(img50, _fb);

    for(auto y = 0 ; y < 5 ; ++y)
        for(auto x = 0 ; x < 5 ; ++x)
            fn(_patchmodels[y*5 + x], extract_corr(feats, 1+8*x, 1+8*y, 16, 16));
}

