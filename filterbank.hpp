#pragma once

#include <vector>

namespace cv {
    class Mat;

    class FilterBank {
    public:
        FilterBank() {};
        FilterBank(const char* fname) { this->load(fname); };

        void load(const char* fname);
        void save(const char* fname) const;

        void add_filter(Mat kernel);
        std::size_t size() const { return _kernels.size(); };

        void filter(const Mat& in, Mat* out_begin) const;
        std::vector<Mat> filter(const Mat& in) const;

    protected:
        std::vector<Mat> _kernels;
    };

} // namespace cv

