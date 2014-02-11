cmake_minimum_required(VERSION 2.8)
project(warco)

if(CMAKE_COMPILER_IS_GNUCXX)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
    string(STRIP "${GCC_VERSION}" GCC_VERSION)
    message(STATUS "GCC Version ${GCC_VERSION} detected")
    message(STATUS "GCC is ${CMAKE_CXX_COMPILER}")

    # GCC below 4.6 does not have C++0x features we use.
    if(${GCC_VERSION} VERSION_LESS "4.6")
        message(FATAL_ERROR "You need at least GCC version 4.6")
    elseif(${GCC_VERSION} VERSION_LESS "4.7")
        add_definitions(-std=c++0x)
    else()
        add_definitions(-std=c++11)
    endif()
endif()

find_package(OpenCV REQUIRED)

set(SRC
    jsoncpp.cpp
    json/json.h
    json/json-forwards.h

    cvutils.cpp
    cvutils.hpp

    covcorr.cpp
    covcorr.hpp
    dists.cpp
    dists.hpp
    features.cpp
    features.hpp

    main.cpp
)

add_executable(warco ${SRC})
target_link_libraries(warco ${OpenCV_LIBS})
