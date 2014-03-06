#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <time.h>

#include <opencv2/core/core.hpp>

#include "covcorr.hpp"
#include "dists.hpp"

int main(int argc, char** argv)
{
    auto seed = argc == 2 ? strtoul(argv[1], nullptr, 0) : time(nullptr);
    std::cout << "Seed is " << seed << std::endl;
    cv::theRNG().state = seed;
    srand(seed);

    warco::test_covcorr();
    warco::test_dists();

    return 0;
}
