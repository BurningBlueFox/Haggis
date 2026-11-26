#include "gpu.h"

#include <emscripten.h>
#include <emscripten/html5.h>

#include <cstdio>
#include <utility>

namespace haggis::web {

void Gpu::connect(std::function<void()> on_ready)
{
    if (!EM_ASM_INT({ return !!navigator.gpu; })) {
        fail("WebGPU is unavailable in this browser.");
        return;
    }

    instance_ = wgpu::CreateInstance();
    instance_.RequestAdapter(nullptr, asynchronous,
        [this, on_ready = std::move(on_ready)](wgpu::RequestAdapterStatus status,
                                               wgpu::Adapter adapter, wgpu::StringView detail) {
            if (status == wgpu::RequestAdapterStatus::Success) {
                request_device(adapter, on_ready);
            } else {
                fail("Could not acquire a WebGPU adapter.", detail);
            }
        });
}

void Gpu::request_device(const wgpu::Adapter& adapter, std::function<void()> on_ready)
{
    const auto descriptor = make_device_descriptor(adapter);
    adapter.RequestDevice(&descriptor, asynchronous,
        [this, adapter, on_ready = std::move(on_ready)](wgpu::RequestDeviceStatus status,
                                                       wgpu::Device device, wgpu::StringView detail) {
            if (status != wgpu::RequestDeviceStatus::Success) {
                fail("Could not create a WebGPU device.", detail);
                return;
            }
            device_ = std::move(device);
            queue_ = device_.GetQueue();
            if (configure_canvas(adapter) && !failed_) {
                on_ready();
            }
        });
}

wgpu::DeviceDescriptor Gpu::make_device_descriptor(const wgpu::Adapter& adapter)
{
    static constexpr auto timestamp = wgpu::FeatureName::TimestampQuery;
    wgpu::DeviceDescriptor descriptor;
    if (adapter.HasFeature(timestamp)) {
        descriptor.requiredFeatures = &timestamp;
        descriptor.requiredFeatureCount = 1;
    }
    descriptor.SetUncapturedErrorCallback(
        +[](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView detail, Gpu* gpu) {
            gpu->fail("WebGPU reported an error. See the browser console.", detail);
        }, this);
    descriptor.SetDeviceLostCallback(asynchronous,
        [this](const wgpu::Device&, wgpu::DeviceLostReason, wgpu::StringView detail) {
            fail("WebGPU device lost. Reload to restart.", detail);
        });
    return descriptor;
}

bool Gpu::configure_canvas(const wgpu::Adapter& adapter)
{
    constexpr auto selector = "#canvas";
    wgpu::EmscriptenSurfaceSourceCanvasHTMLSelector canvas;
    canvas.selector = selector;
    const wgpu::SurfaceDescriptor descriptor{.nextInChain = &canvas};
    surface_ = instance_.CreateSurface(&descriptor);

    wgpu::SurfaceCapabilities capabilities;
    if (!surface_.GetCapabilities(adapter, &capabilities) || capabilities.formatCount == 0) {
        fail("No compatible canvas format.");
        return false;
    }

    int width = 0, height = 0;
    emscripten_get_canvas_element_size(selector, &width, &height);
    format_ = capabilities.formats[0];
    const wgpu::SurfaceConfiguration config{
        .device = device_, .format = format_,
        .width = static_cast<std::uint32_t>(width), .height = static_cast<std::uint32_t>(height),
    };
    surface_.Configure(&config);
    return true;
}

void Gpu::fail(const char* reason, wgpu::StringView detail)
{
    if (std::exchange(failed_, true)) return;
    std::puts(reason); // The HTML shell displays stdout in the status element.
    if (detail.data) {
        std::fprintf(stderr, "%.*s\n", static_cast<int>(detail.length), detail.data);
    }
}

wgpu::Texture Gpu::canvas_texture() const
{
    wgpu::SurfaceTexture frame;
    surface_.GetCurrentTexture(&frame);
    return std::move(frame.texture);
}

wgpu::Buffer Gpu::buffer(wgpu::BufferUsage usage, std::uint64_t bytes) const
{
    const wgpu::BufferDescriptor descriptor{.usage = usage, .size = bytes};
    return device_.CreateBuffer(&descriptor);
}

wgpu::ShaderModule Gpu::shader(std::string_view code) const
{
    wgpu::ShaderSourceWGSL source;
    source.code = code;
    const wgpu::ShaderModuleDescriptor descriptor{.nextInChain = &source};
    return device_.CreateShaderModule(&descriptor);
}

}
