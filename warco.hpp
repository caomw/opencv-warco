#pragma once

#include <vector>
#include <memory>

namespace cv {
    class Mat;
}

namespace warco {

    struct PatchModel;

    struct Warco {

        Warco();
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

        void foreach_model(const cv::Mat& img, std::function<void(const Patch& patch, const cv::Mat& corr)> fn) const;
    };

} // namespace warco

