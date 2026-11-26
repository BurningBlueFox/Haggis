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

## Browser build

Install and activate Emscripten 4.0.23 with [emsdk](https://emscripten.org/docs/getting_started/downloads.html), then configure a separate build directory.

```sh
emcmake cmake -S . -B build-web -DCMAKE_BUILD_TYPE=Release
cmake --build build-web --parallel
python3 -m http.server --directory build-web/web 8000
```

Open http://localhost:8000 to see the Haggis canvas and startup message.

GitHub Actions builds the browser target for pull requests and deploys it on pushes to `main`. In the repository's **Settings → Pages**, set **Source** to **GitHub Actions** to enable deployment at https://burningbluefox.github.io/Haggis/.
