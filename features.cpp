#include "features.hpp"

#include <opencv2/opencv.hpp>

#ifndef NDEBUG
#include <iostream>
#include "to_s.hpp"
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
    std::cout << "g1: " << to_s(g1) << " ; g2: " << to_s(g2) << std::endl;
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
    std::cout << "g: " << to_s(g) << " ; G: " << to_s(G) << std::endl;
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
    Features nrvo(3+8);

#ifndef NDEBUG
    std::cout << "m: " << to_s(m) << std::endl;
#endif

    // Get L*a*b* values out of it. They are all in [0,255] range since m is U8.
    cv::Mat lab;
    cvtColor(m, lab, CV_BGR2Lab);
#ifndef NDEBUG
    std::cout << "L*a*b*: " << to_s(lab) << std::endl;
#endif

    // But we work with float matrices only!
    // Might go for signed 16bit at some point, but only as an optimization if
    // needed since we need to be careful with computations.
    cv::Mat labf;
    lab.convertTo(labf, CV_32FC3, 1./255.);
    split(labf, &nrvo[0]);

#ifndef NDEBUG
    std::cout << "L*: " << to_s(nrvo[0]) << " ; a*: " << to_s(nrvo[1]) << " ; b*: " << to_s(nrvo[2]) << std::endl;
#endif

    mkDooG(nrvo[0], &nrvo[3]);

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

