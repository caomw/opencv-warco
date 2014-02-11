#include "dists.hpp"

#include <stdexcept>
#include <opencv2/opencv.hpp>

#include "cvutils.hpp"

static cv::Mat logp_id(const cv::Mat& m)
{
    return warco::eig_fn(m, [](double lambda) { return log(lambda); });
}

static float euc_sq(const cv::Mat& lA, cv::Mat& lB)
{
    auto d = lB - lA;
    return trace(d*d)[0];
}

float warco::dist_euc(const cv::Mat& corrA, const cv::Mat& corrB)
{
    // TODO: could hoist this, since we'll have duplicate work here.
    cv::Mat lA = logp_id(corrA);
    cv::Mat lB = logp_id(corrB);

    return sqrt(euc_sq(lA, lB));
}

static void test_euc()
{
    std::cout << "Euclidean distance... " << std::flush;
    cv::Mat A = warco::randspd(4,4),
            B = warco::randspd(4,4);

    if(warco::dist_euc(A, A) > 1e-6) {
        std::cerr << "Failed (d(A,A) is not close to 0)!" << std::endl;
        throw std::runtime_error("Test assertion failed.");
    }

    if(std::abs(warco::dist_euc(A, B) - warco::dist_euc(B, A)) > 1e-6) {
        std::cerr << "Failed (d(A,B) is not close to d(B,A))!" << std::endl;
        throw std::runtime_error("Test assertion failed.");
    }

    std::cout << "SUCCESS" << std::endl;
}

float warco::dist_cbh(const cv::Mat& corrA, const cv::Mat& corrB)
{
    // TODO: could hoist this, since we'll have duplicate work here.
    cv::Mat lA = logp_id(corrA);
    cv::Mat lB = logp_id(corrB);

    cv::Mat d = lB - lA;
    float E = trace(d*d)[0];

    cv::Mat ab = lA * lB,
            ab2 = ab*ab,
            a2 = lA * lA,   // This and below could be taken out too.
            b2 = lB * lB,   // Or cached.
            a2b2 = a2 * b2;
    float xi = -1./12. * (trace(ab2)[0] - trace(a2b2)[0]);

    return sqrt(E + xi);
}

static void test_cbh()
{
    std::cout << "CBH distance... " << std::flush;
    cv::Mat A = warco::randspd(4,4),
            B = warco::randspd(4,4);

    if(warco::dist_cbh(A, A) > 1e-6) {
        std::cerr << "Failed (d(A,A) is not close to 0)!" << std::endl;
        throw std::runtime_error("Test assertion failed.");
    }

    if(std::abs(warco::dist_cbh(A, B) - warco::dist_cbh(B, A)) > 1e-6) {
        std::cerr << "Failed (d(A,B) is not close to d(B,A))!" << std::endl;
        throw std::runtime_error("Test assertion failed.");
    }

    std::cout << "SUCCESS" << std::endl;
}

void warco::test_dists()
{
    test_euc();
    test_cbh();
}

