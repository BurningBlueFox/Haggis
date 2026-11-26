# Haggis
Haggis is a C++ project game / simulation engine in progress with gpu inference built in

## Build and run

Install CMake 4.5.1 or newer and a compiler with C++26 mode (GCC 14+ or Clang 17+), then run these commands from the repository root on Linux or macOS.

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

## Browser build

Install and activate Emscripten 4.0.23 with [emsdk](https://emscripten.org/docs/getting_started/downloads.html), then configure a separate build directory.

```sh
emcmake cmake -S . -B build-web -DCMAKE_BUILD_TYPE=Release
cmake --build build-web --parallel
python3 -m http.server --directory build-web/web 8000
```

Open http://localhost:8000 in a WebGPU-capable browser. The triangle rotates while a continuous WGSL compute workload advances four-component vectors through 32 fixed-weight recurrent steps. The render pass uses the most recently completed color. Uncheck **Run inference workload** to measure rendering on its own.

The frame loop starts at 120 FPS and uses observed display cadence, frame intervals, and render cost to select 60, 120, or 240 FPS. While compute is enabled it accumulates 10% of frame time as credit and lends spare time to compute when frames are on pace. Dispatch size adapts to observed cost. One compute chunk can be outstanding, and its completion arrives asynchronously. If the smallest workgroup exceeds its budget, the scheduler carries credit across frames.

The page reports p95 frame interval, vectors completed per second, and compute time as a percentage of the measurement window. Adapters with timestamp queries measure GPU passes directly. Other adapters use asynchronous queue completion latency and periodic render-only samples to estimate cost. Achievable frame rate and compute share depend on the GPU and browser.

GitHub Actions builds the browser target for pull requests and deploys it on pushes to `main`. In the repository's **Settings → Pages**, set **Source** to **GitHub Actions** to enable deployment at https://burningbluefox.github.io/Haggis/.
