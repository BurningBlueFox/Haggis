"""Exercise the deployed Wasm entry point with Chromium's software WebGPU adapter."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
import re
import sys
from threading import Thread

from playwright.sync_api import expect, sync_playwright


def verify_delayed_timestamp_readback(page):
    # Hold one mapped result while the animation loop keeps drawing.
    page.evaluate("""() => {
        const state = globalThis.timestampProbe = { submissions: 0, release: null };
        const submit = GPUQueue.prototype.submit;
        const mapAsync = GPUBuffer.prototype.mapAsync;
        GPUQueue.prototype.submit = function (...args) {
            state.submissions++;
            return submit.apply(this, args);
        };
        GPUBuffer.prototype.mapAsync = function (...args) {
            const result = mapAsync.apply(this, args);
            return result.then(value => {
                GPUBuffer.prototype.mapAsync = mapAsync;
                return new Promise(resolve => {
                    state.release = () => resolve(value);
                });
            });
        };
    }""")
    page.wait_for_function("timestampProbe.release !== null")
    submissions = page.evaluate("timestampProbe.submissions")
    try:
        page.wait_for_function("count => timestampProbe.submissions >= count", arg=submissions + 8)
    finally:
        page.evaluate("timestampProbe.release()")


def exercise(browser, site, *, timestamps):
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    if not timestamps:
        # Advertise the same adapter without its optional timestamp feature.
        page.add_init_script("""
            const features = Object.getOwnPropertyDescriptor(GPUAdapter.prototype, 'features').get;
            Object.defineProperty(GPUAdapter.prototype, 'features', {get() {
                const available = new Set(features.call(this));
                available.delete('timestamp-query');
                return available;
            }});
        """)
    page.goto(site)
    status = page.locator("#status")
    metrics = page.locator("#metrics")
    expect(status).to_have_text("WebGPU ready")
    expect(metrics).to_contain_text("GPU timestamps" if timestamps else "Queue latency estimate")
    expect(metrics).to_contain_text(re.compile(r"\b[1-9]\d* vectors/s"))
    page.locator("#compute").uncheck()
    expect(metrics).to_contain_text(re.compile(r"\b0 vectors/s"))
    if timestamps:
        verify_delayed_timestamp_readback(page)
    canvas = page.locator("#canvas")
    first = canvas.screenshot()
    # Rendering continues after compute pauses. Two distinct images verify motion.
    for _ in range(5):
        if canvas.screenshot() != first:
            break
    else:
        raise AssertionError("Canvas did not animate")
    page.locator("#compute").check()
    expect(metrics).to_contain_text(re.compile(r"\b[1-9]\d* vectors/s"))
    expect(status).to_have_text("WebGPU ready")
    assert not errors, errors
    page.close()


if __name__ == "__main__":
    files = partial(SimpleHTTPRequestHandler, directory=sys.argv[1])
    with ThreadingHTTPServer(("127.0.0.1", 0), files) as server, sync_playwright() as playwright:
        Thread(target=server.serve_forever, daemon=True).start()
        site = f"http://127.0.0.1:{server.server_port}/"
        browser = playwright.chromium.launch(
            executable_path=os.getenv("CHROME_BIN"),
            args=["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan=swiftshader",
                  "--use-webgpu-adapter=swiftshader", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
        )
        expect.set_options(timeout=15000)
        exercise(browser, site, timestamps=True)
        exercise(browser, site, timestamps=False)
        browser.close()
