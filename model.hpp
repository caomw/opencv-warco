#pragma once

#include <vector>
#include <memory>

namespace cv {
    class Mat;
}

struct svm_model;
struct svm_problem;

namespace warco {

    void test_model();

    struct PatchModel {
        using Ptr = std::shared_ptr<PatchModel>;

        PatchModel();
        ~PatchModel();

        void add_sample(const cv::Mat& corr, unsigned label);
        double train();
        unsigned predict(const cv::Mat& corr) const;
        std::vector<double> predict_probas(const cv::Mat& corr) const;

        void save(const char* name) const;
        void load(const char* name);

    protected:
        std::vector<cv::Mat> _corrs;
        std::vector<double> _lbls;
        svm_model* _svm;
        svm_problem* _prob;
        double _mean;

        void free_svm();
    };

    struct Warco {
        Warco() {};
        ~Warco() {};

    protected:
        std::vector<std::tuple<double, PatchModel::Ptr>> _patchmodels;
        std::vector<std::string> _lbls;
    };

} // namespace warco

