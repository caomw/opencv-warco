#include "features.hpp"

#include <opencv2/opencv.hpp>

// For CV_BGR2Lab, at least in opencv trunk.
#include <opencv2/imgproc/types_c.h>

#ifndef NDEBUG
#include <iostream>
#include "to_s.hpp"
using warco::to_s;
#endif

static void mkDooG(const cv::Mat& l, cv::Mat* out)
{
    const int w = l.cols, h = l.rows;

    // Blur with two gaussians, used for DooG computation further down.
    const double sigma1 = 0.5;
    const double sigma2 = 1.0;
    cv::Mat g1, g2;
    GaussianBlur(l, g1, cv::Size(3,3), sigma1, sigma1, cv::BORDER_REFLECT_101);
    GaussianBlur(l, g2, cv::Size(3,3), sigma2, sigma2, cv::BORDER_REFLECT_101);

#ifndef NDEBUG
    if(getenv("WARCO_DEBUG")) {
        std::cout << "g1: " << to_s(g1) << " ; g2: " << to_s(g2) << std::endl;
    }
#endif

    const int top    = 2,
              bottom = 2,
              left   = 2,
              right  = 2;
    cv::Mat g, G;
    copyMakeBorder(g1, g, top, bottom, left, right, cv::BORDER_REFLECT_101);
    copyMakeBorder(g2, G, top, bottom, left, right, cv::BORDER_REFLECT_101);
    // Lol this implies double-mirror on the borders, careful!

#ifndef NDEBUG
    if(getenv("WARCO_DEBUG")) {
        std::cout << "g: " << to_s(g) << " ; G: " << to_s(G) << std::endl;
    }
#endif

    auto mkr = [w,h](int l, int t) { return cv::Rect(l, t, w, h); };

    *out++ = g(mkr(left, 4)) - 2.0*g(mkr(left, 2)) + g(mkr(left, 0));
    *out++ = g(mkr(left, 1))                       - g(mkr(left, 3));
    *out++ = g(mkr(4,  top)) - 2.0*g(mkr(2,  top)) + g(mkr(0,  top));
    *out++ = g(mkr(1,  top))                       - g(mkr(3,  top));

    *out++ = G(mkr(left, 4)) - 2.0*G(mkr(left, 2)) + G(mkr(left, 0));
    *out++ = G(mkr(left, 1))                       - G(mkr(left, 3));
    *out++ = G(mkr(4,  top)) - 2.0*g(mkr(2,  top)) + G(mkr(0,  top));
    *out++ = G(mkr(1,  top))                       - G(mkr(3,  top));
}

warco::Features warco::mkfeats(const cv::Mat& m)
{
    // Layout is:
    // 0-2: L, a, b
    // 3-6: 4 "sharp" DooG gradients
    // 7-10: 4 "smooth" DooG gradients
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
    lab.convertTo(labf, CV_32FC3, 1./255.);
    split(labf, &nrvo[0]);

#ifndef NDEBUG
    if(getenv("WARCO_DEBUG")) {
        std::cout << "L*: " << to_s(nrvo[0]) << " ; a*: " << to_s(nrvo[1]) << " ; b*: " << to_s(nrvo[2]) << std::endl;
    }
#endif

    // Compute the DooG filtered version into 3-10
    mkDooG(nrvo[0], &nrvo[3]);

    // Compute the gradient mag/ori into 11-12
    cv::Mat dx, dy;
    const int ksize = 1;
    Sobel(nrvo[0], dx, CV_32F, 1, 0, ksize);
    Sobel(nrvo[0], dy, CV_32F, 0, 1, ksize);
    magnitude(dx, dy, nrvo[11]);
    phase(dx, dy, nrvo[12]); // in radians by default.

    return nrvo;
}

void warco::showfeats(const Features& imgs)
{
    for(auto img : imgs) {
        cv::Mat normalized;
        normalize(img, normalized, 0.0, 1.0, CV_MINMAX);
        imshow("Feature", normalized);
        cv::waitKey(0);
    }
}

