#include "dists.hpp"

#include <stdexcept>
#include <opencv2/opencv.hpp>

#include "cvutils.hpp"

// Randomly generated, but computed using original matlab implementation.
static cv::Mat wA = (cv::Mat_<double>(4,4) <<
    4.6503810, -0.8571707,  0.5054186, -0.6182507,
    -0.8571707,  3.5171059,  0.2891554,  1.4221318,
    0.5054186,  0.2891554,  3.6702419, -0.2359455,
    -0.6182507,  1.4221318, -0.2359455,  4.1547181
);
static cv::Mat wB = (cv::Mat_<double>(4,4) <<
    5.8768365,  2.1580387, -2.0379778,  2.1673869,
    2.1580387,  9.7772208, -0.2766553, -1.1106845,
    -2.0379778, -0.2766553,  7.6452496,  0.0017575,
    2.1673869, -1.1106845,  0.0017575, 11.1753981
);

warco::distfn_t warco::get_distfn(const std::string& name)
{
    if(name == "euclid") {
        return dist_euc;
    } else if(name == "cbh") {
        return dist_cbh;
    } else if(name == "geodesic") {
        return dist_geo;
    } else if(name == "my euclid") {
        return dist_my_euc;
    } else {
        throw std::runtime_error("Unknown distance function: '" + name + "'");
    }
}

std::string warco::get_name(distfn_t fn)
{
    const auto* pfn = fn.target<float(*)(const cv::Mat&, const cv::Mat&)>();

    if(pfn && *pfn == dist_euc) {
        return "euclid";
    } else if(pfn && *pfn == dist_cbh) {
        return "cbh";
    } else if(pfn && *pfn == dist_geo) {
        return "geodesic";
    } else if(pfn && *pfn == dist_my_euc) {
        return "my euclid";
    } else {
        throw std::runtime_error("unknown distance function in use!");
    }
}

static cv::Mat logp_id(const cv::Mat& m)
{
    return warco::eig_fn(m, [](double lambda) { return log(lambda); });
}

static void test_logp_id()
{
    std::cout << "logp_id... " << std::flush;

    warco::assert_mat_almost_eq(logp_id(wA), (cv::Mat_<double>(4,4) <<
        1.5033978, -0.2045823,  0.1300535, -0.1046505,
       -0.2045823,  1.1510779,  0.1123214,  0.3851847,
        0.1300535,  0.1123214,  1.2840035, -0.0731576,
       -0.1046505,  0.3851847, -0.0731576,  1.3458774
    ));

    warco::assert_mat_almost_eq(logp_id(wB), (cv::Mat_<double>(4,4) <<
        1.6013708,  0.3237514, -0.3289237,  0.3071247,
        0.3237514,  2.2303997,  0.0127973, -0.1470679,
       -0.3289237,  0.0127973,  1.9869731,  0.0403853,
        0.3071247, -0.1470679,  0.0403853,  2.3704253
    ));

    std::cout << "SUCCESS" << std::endl;
}

static float euc_sq(const cv::Mat& lA, const cv::Mat& lB)
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
    using warco::reldiff;

    std::cout << "Euclidean distance... " << std::flush;
    cv::Mat A = warco::randspd(4,4),
            B = warco::randspd(4,4);

    double dAA = warco::dist_euc(A, A);
    if(dAA > 1e-6) {
        std::cerr << "Failed! (d(A,A)=" << dAA << " is not close to 0)" << std::endl;
        throw std::runtime_error("Test assertion failed.");
    }

    double dAB = warco::dist_euc(A, B);
    double dBA = warco::dist_euc(B, A);
    if(reldiff(dAB, dBA) > 1e-6) {
        std::cerr << "Failed! (rel diff (dAB, dBA) = " << reldiff(dAB, dBA) << " is too large)" << std::endl;
        throw std::runtime_error("Test assertion failed.");
    }

    double dI1 = warco::dist_cbh(cv::Mat::eye(4, 4, CV_32F), 2*cv::Mat::eye(4, 4, CV_32F));
    if(reldiff(dI1, 1.386294) > 1e-6) {
        std::cerr << "Failed! (rel diff (dI1, 1.386294) = " << reldiff(dI1, 1.386294) << " is too large)" << std::endl;
        throw std::runtime_error("Test assertion failed.");
    }

    double dwAwB = warco::dist_euc(wA, wB);
    if(reldiff(dwAwB, 2.156221) > 1e-6) {
        std::cerr << "Failed! (rel diff (dwAwB=" << dwAwB << ", 2.156221) = " << reldiff(dwAwB, 2.156221) << " is too large)" << std::endl;
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
    using warco::reldiff;

    std::cout << "CBH distance... " << std::flush;
    cv::Mat A = warco::randspd(4,4),
            B = warco::randspd(4,4);

    double dAA = warco::dist_cbh(A, A);
    if(dAA > 1e-6) {
        std::cerr << "Failed! (d(A,A)=" << dAA << " is not close to 0)" << std::endl;
        throw std::runtime_error("Test assertion failed.");
    }

    double dAB = warco::dist_cbh(A, B);
    double dBA = warco::dist_cbh(B, A);
    if(reldiff(dAB, dBA) > 1e-6) {
        std::cerr << "Failed! (rel diff (dAB, dBA) = " << reldiff(dAB, dBA) << " is too large)" << std::endl;
        throw std::runtime_error("Test assertion failed.");
    }

    double dI1 = warco::dist_cbh(cv::Mat::eye(4, 4, CV_32F), 2*cv::Mat::eye(4, 4, CV_32F));
    if(reldiff(dI1, 1.386294) > 1e-6) {
        std::cerr << "Failed! (rel diff (dI1, 1.386294) = " << reldiff(dI1, 1.386294) << " is too large)" << std::endl;
        throw std::runtime_error("Test assertion failed.");
    }

    double dwAwB = warco::dist_cbh(wA, wB);
    if(reldiff(dwAwB, 2.156895) > 1e-6) {
        std::cerr << "Failed! (rel diff (dwAwB=" << dwAwB << ", 2.156895) = " << reldiff(dwAwB, 2.156895) << " is too large)" << std::endl;
        throw std::runtime_error("Test assertion failed.");
    }

    std::cout << "SUCCESS" << std::endl;
}

float warco::dist_geo(const cv::Mat& corrA, const cv::Mat& corrB)
{
    // TODO
    // Weird, from both the paper and logic these should not involve logp_id,
    // but from the code, they do.
    //cv::Mat lA = logp_id(corrA);
    //cv::Mat lB = logp_id(corrB);
    cv::Mat lA = corrA;
    cv::Mat lB = corrB;

    cv::Mat lA_inv_sqrt = warco::eig_fn(lA, [](double lambda) {
        return 1./sqrt(lambda);
    });

    // I'm sure this `thingy` has a meaning.
    cv::Mat thingy = logp_id(lA_inv_sqrt * lB * lA_inv_sqrt);

    // NOTE: the sqrt is missing in tosato's code in Y_GSVM_Train_deterministic:47
    //       (He doesn't seem to ever execute that part anyways)
    return sqrt(trace(thingy*thingy)[0]);
}

static void test_geo()
{
    using warco::reldiff;

    std::cout << "Geodesic distance... " << std::flush;
    cv::Mat A = warco::randspd(4,4),
            B = warco::randspd(4,4);

    // TODO: Is it not a bug that geodesic is much more sensitive?
    double dAA = warco::dist_geo(A, A);
    if(dAA > 1e-5) {
        std::cerr << "Failed! (d(A,A)=" << dAA << " is not close to 0)" << std::endl;
        throw std::runtime_error("Test assertion failed.");
    }

    // TODO: geo definitely isn't symmetric numerically.
    double dAB = warco::dist_geo(A, B);
    double dBA = warco::dist_geo(B, A);
    if(reldiff(dAB, dBA) > 1e-5) {
        std::cerr << "Failed! (rel diff (dAB, dBA) = " << reldiff(dAB, dBA) << " is too large)" << std::endl;
        throw std::runtime_error("Test assertion failed.");
    }

    double dI1 = warco::dist_geo(cv::Mat::eye(4, 4, CV_32F), 2*cv::Mat::eye(4, 4, CV_32F));
    if(reldiff(dI1, 1.386294) > 1e-6) {
        std::cerr << "Failed! (rel diff (dI1, 1.386294) = " << reldiff(dI1, 1.386294) << " is too large)" << std::endl;
        throw std::runtime_error("Test assertion failed.");
    }

    double dwAwB = warco::dist_geo(wA, wB);
    if(reldiff(dwAwB, 2.1575107) > 1e-6) {
        std::cerr << "Failed! (rel diff (dwAwB=" << dwAwB << ", 2.1575107) = " << reldiff(dwAwB, 2.1575107) << " is too large)" << std::endl;
        throw std::runtime_error("Test assertion failed.");
    }

    std::cout << "SUCCESS" << std::endl;
}

float warco::dist_my_euc(const cv::Mat& corrA, const cv::Mat& corrB)
{
    return sqrt(euc_sq(corrA, corrB));
}

static void test_my_euc()
{
    using warco::reldiff;

    std::cout << "[My] Euclidean distance... " << std::flush;
    cv::Mat A = warco::randspd(4,4),
            B = warco::randspd(4,4);

    double dAA = warco::dist_my_euc(A, A);
    if(dAA > 1e-6) {
        std::cerr << "Failed! (d(A,A)=" << dAA << " is not close to 0)" << std::endl;
        throw std::runtime_error("Test assertion failed.");
    }

    double dAB = warco::dist_my_euc(A, B);
    double dBA = warco::dist_my_euc(B, A);
    if(reldiff(dAB, dBA) > 1e-6) {
        std::cerr << "Failed! (rel diff (dAB, dBA) = " << reldiff(dAB, dBA) << " is too large)" << std::endl;
        throw std::runtime_error("Test assertion failed.");
    }

    std::cout << "SUCCESS" << std::endl;
}

void warco::test_dists()
{
    test_logp_id();

    test_euc();
    test_cbh();
    test_geo();

    test_my_euc();
}

