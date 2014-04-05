#include "features.hpp"

#include <opencv2/opencv.hpp>
// For CV_BGR2Lab, at least in opencv trunk.
#include <opencv2/imgproc/types_c.h>

#include "filterbank.hpp"

#ifndef NDEBUG
#  include <iostream>
#  include "to_s.hpp"
#endif

warco::Features warco::mkfeats(const cv::Mat& m, const cv::FilterBank& fb)
{
    // Layout is (inclusive):
    // 0-3: 4 "sharp" DooG gradients
    // 4-7: 4 "smooth" DooG gradients
    // 8-10: L, a, b
    // 11: gradient magnitude
    // 12: gradient orientation
    Features nrvo(3+8+2);

#ifndef NDEBUG
    if(getenv("WARCO_DEBUG")) {
        std::cout << "m: " << to_s(m) << std::endl;
    }
#endif

    // Get L*a*b* values out of it. They are all in [0,255] range since m is U8.
    cv::Mat lab;
    cvtColor(m, lab, CV_BGR2Lab);
#ifndef NDEBUG
    if(getenv("WARCO_DEBUG")) {
        std::cout << "L*a*b*: " << to_s(lab) << std::endl;
    }
#endif

    // But we work with float matrices only!
    // Might go for signed 16bit at some point, but only as an optimization if
    // needed since we need to be careful with computations.
    cv::Mat labf;
    lab.convertTo(labf, CV_32FC3);
    split(labf, &nrvo[8]);

    const cv::Mat& l = nrvo[8];

#ifndef NDEBUG
    if(getenv("WARCO_DEBUG")) {
        std::cout << "L*: " << to_s(l) << " ; a*: " << to_s(nrvo[9]) << " ; b*: " << to_s(nrvo[10]) << std::endl;
    }
#endif

    // Compute the DooG filtered version into 0-7
    fb.filter(l, &nrvo[0]);

    // Compute the gradient mag/ori into 11-12
    cv::Mat dx, dy;
    const int ksize = 1;
    Sobel(l, dx, CV_32F, 1, 0, ksize);
    Sobel(l, dy, CV_32F, 0, 1, ksize);
    magnitude(dx, dy, nrvo[11]);
    phase(dx, dy, nrvo[12]); // in radians [0,2pi] by default.

    // The following makes 30 be the same as 210 degree.
    // Interestingly, the results stay exactly the same.
#if 0
    for(int y = 0 ; y < nrvo[12].rows ; ++y) {
        float* line = nrvo[12].ptr<float>(y);
        for(int x = 0 ; x < nrvo[12].cols ; ++x, ++line)
            if(*line > M_PI)
                *line -= M_PI;
    }
#endif

    return nrvo;
}

void warco::showfeats(const Features& feats)
{
    for(auto& feat : feats) {
        cv::Mat normalized;
        normalize(feat, normalized, 0.0, 1.0, CV_MINMAX);
        imshow("Feature", normalized);
        cv::waitKey(0);
    }
}

