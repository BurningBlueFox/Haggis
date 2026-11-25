cmake_minimum_required(VERSION 4.5.1)
project(Haggis LANGUAGES CXX)

add_executable(haggis src/main.cpp)

include(CTest)

if(BUILD_TESTING)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://codeload.github.com/google/googletest/tar.gz/refs/tags/v1.18.0
        URL_HASH SHA256=6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5
        EXCLUDE_FROM_ALL
    )
    set(BUILD_GMOCK OFF)
    set(INSTALL_GTEST OFF)
    FetchContent_MakeAvailable(googletest)

    add_executable(haggis_tests tests/greeting_test.cpp)
    target_include_directories(haggis_tests PRIVATE src)
    target_link_libraries(haggis_tests PRIVATE GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(haggis_tests)
endif()
