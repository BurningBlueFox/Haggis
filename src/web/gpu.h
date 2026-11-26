#pragma once

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <functional>
#include <string_view>

namespace haggis::web {

inline constexpr auto asynchronous = wgpu::CallbackMode::AllowSpontaneous;

class Gpu {
public:
    void connect(std::function<void()> on_ready);
    void fail(const char* reason, wgpu::StringView detail = {});

    [[nodiscard]] bool failed() const { return failed_; }
    [[nodiscard]] const wgpu::Device& device() const { return device_; }
    [[nodiscard]] const wgpu::Queue& queue() const { return queue_; }
    [[nodiscard]] wgpu::TextureFormat canvas_format() const { return format_; }
    [[nodiscard]] wgpu::Texture canvas_texture() const;

    [[nodiscard]] wgpu::Buffer buffer(wgpu::BufferUsage usage, std::uint64_t bytes) const;
    [[nodiscard]] wgpu::ShaderModule shader(std::string_view source) const;

private:
    void request_device(const wgpu::Adapter& adapter, std::function<void()> on_ready);
    bool configure_canvas(const wgpu::Adapter& adapter);

    wgpu::Instance instance_;
    wgpu::Surface surface_;
    wgpu::Device device_;
    wgpu::Queue queue_;
    wgpu::TextureFormat format_ = wgpu::TextureFormat::Undefined;
    bool failed_ = false;
};

}
