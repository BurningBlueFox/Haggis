cmake_minimum_required(VERSION 4.5.1)
project(Haggis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(EMSCRIPTEN)
    add_executable(haggis
        src/web_main.cpp
        src/web/app.cpp
        src/web/gpu.cpp
        src/web/gpu_debug.cpp
        src/web/gpu_timing.cpp
        src/web/inference.cpp
        src/web/triangle.cpp
    )
    target_include_directories(haggis PRIVATE src)
    target_compile_options(haggis PRIVATE --use-port=emdawnwebgpu)
    target_link_options(haggis PRIVATE --use-port=emdawnwebgpu)
    set(web_shell "${CMAKE_CURRENT_SOURCE_DIR}/web/shell.html")
    set(web_output "${CMAKE_CURRENT_BINARY_DIR}/web")
    set(HAGGIS_WEB_REVISION "" CACHE STRING "Revision for versioned deployment assets")
    set_target_properties(haggis PROPERTIES
        OUTPUT_NAME index
        SUFFIX ".html"
        RUNTIME_OUTPUT_DIRECTORY "${web_output}"
        LINK_DEPENDS "${web_shell}"
    )
    target_link_options(haggis PRIVATE "SHELL:--shell-file \"${web_shell}\"")
    if(HAGGIS_WEB_REVISION)
        # An old loader or Wasm binary must never be reused for a new page.
        set(bundle_name "haggis-${HAGGIS_WEB_REVISION}")
        set(bundle_dir "${CMAKE_CURRENT_BINARY_DIR}/web-bundle")
        set_target_properties(haggis PROPERTIES
            OUTPUT_NAME "${bundle_name}"
            RUNTIME_OUTPUT_DIRECTORY "${bundle_dir}"
        )
        add_custom_command(TARGET haggis POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${web_output}"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${bundle_dir}/${bundle_name}.js"
                "${bundle_dir}/${bundle_name}.wasm"
                "${web_output}"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "$<TARGET_FILE:haggis>" "${web_output}/index.html"
            VERBATIM
        )
    endif()
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
