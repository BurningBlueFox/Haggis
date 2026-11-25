# Haggis
Haggis is a C++ project game / simulation engine in progress with gpu inference built in

## Build and run

Install CMake 4.5.1 or newer and a C++ compiler, then run these commands from the repository root on Linux or macOS.

```sh
cmake -S . -B build
cmake --build build
./build/haggis
```

## Run tests

The default build downloads GoogleTest 1.18.0 and builds the unit tests. Run them with:

```sh
ctest --test-dir build --output-on-failure
```

Configure with `-DBUILD_TESTING=OFF` to build only the executable without downloading GoogleTest.
