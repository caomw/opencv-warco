#pragma once

#include <functional>
#include <memory>
#include <vector>

// For FilterBank.
// TODO: Maybe keep a unique pointer so fwd decl is enough?
#include "filterbank.hpp"

namespace cv {
    class Mat;
}

namespace warco {

    struct PatchModel;

    struct Warco {

        Warco(cv::FilterBank fb);
        ~Warco();

        void add_sample(const cv::Mat& img, unsigned label);

        double train();
        unsigned predict(const cv::Mat& img) const;
        unsigned predict_proba(const cv::Mat& img) const;

        // TODO
        void save(const char* name) const;
        void load(const char* name);

    protected:
        struct Patch {
            double w;
            std::unique_ptr<PatchModel> model;

            Patch();
        };

        std::vector<Patch> _patchmodels;

        cv::FilterBank _fb;

        void foreach_model(const cv::Mat& img, std::function<void(const Patch& patch, const cv::Mat& corr)> fn) const;
    };

} // namespace warco

