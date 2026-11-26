cmake_minimum_required(VERSION 4.5.1)
project(Haggis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(EMSCRIPTEN)
    add_executable(haggis src/web_main.cpp)
    target_compile_options(haggis PRIVATE --use-port=emdawnwebgpu)
    target_link_options(haggis PRIVATE --use-port=emdawnwebgpu)
    set(web_shell "${CMAKE_CURRENT_SOURCE_DIR}/web/shell.html")
    set_target_properties(haggis PROPERTIES
        OUTPUT_NAME index
        SUFFIX ".html"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/web"
        LINK_DEPENDS "${web_shell}"
    )
    target_link_options(haggis PRIVATE "SHELL:--shell-file \"${web_shell}\"")
    return()
endif()

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

    add_executable(haggis_tests tests/greeting_test.cpp tests/frame_budget_test.cpp)
    target_include_directories(haggis_tests PRIVATE src)
    target_link_libraries(haggis_tests PRIVATE GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(haggis_tests)
endif()
