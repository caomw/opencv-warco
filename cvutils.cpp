#include "cvutils.hpp"

#include <opencv2/opencv.hpp>

cv::Mat warco::eig_fn(const cv::Mat& m, std::function<double (double)> fn)
{
    cv::Mat eigvals;
    cv::Mat eigvecs;

    if(! eigen(m, eigvals, eigvecs))
        throw std::runtime_error("Cannot eigen-decompose matrix.");

    for(auto eig = eigvals.begin<float>() ; eig != eigvals.end<float>() ; ++eig)
        *eig = fn(*eig);

    return eigvecs * cv::Mat::diag(eigvals) * eigvecs.t();
}

void warco::assert_mat_almost_eq(const cv::Mat& actual, const cv::Mat& expected)
{
    auto maxdiff = norm(actual - expected, cv::NORM_INF);
    if(maxdiff > 1e-6) {
        std::cerr << "FAILED (max diff: " << maxdiff << ")" << std::endl;
        std::cerr << "Expected:" << std::endl;
        std::cerr << expected << std::endl;
        std::cerr << "actual:" << std::endl;
        std::cerr << actual << std::endl;
        throw std::runtime_error("Test assertion failed.");
    }
}

