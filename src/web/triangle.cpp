#include "triangle.h"

#include <array>

namespace haggis::web {
namespace {

constexpr std::string_view shader = R"(
@group(0) @binding(0) var<uniform> time: f32;
@group(0) @binding(1) var<uniform> color: vec4f;

@vertex fn vertex_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let vertices = array(vec2f(0, .63), vec2f(-.55, -.32), vec2f(.55, -.32));
    let c = cos(time);
    let s = sin(time);
    let p = mat2x2f(vec2f(c, s), vec2f(-s, c)) * vertices[i];
    return vec4f(p.x * 9.0 / 16.0, p.y, 0, 1);
}
@fragment fn fragment_main() -> @location(0) vec4f {
    return color;
}
)";

}

void Triangle::initialize(const Gpu& gpu, const wgpu::Buffer& color)
{
    const auto module = gpu.shader(shader);
    const wgpu::ColorTargetState target{.format = gpu.canvas_format()};
    const wgpu::FragmentState fragment{
        .module = module, .entryPoint = "fragment_main", .targetCount = 1, .targets = &target,
    };
    const wgpu::RenderPipelineDescriptor descriptor{
        .vertex = {.module = module, .entryPoint = "vertex_main"}, .fragment = &fragment,
    };
    pipeline_ = gpu.device().CreateRenderPipeline(&descriptor);

    rotation_ = gpu.buffer(wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, sizeof(float));
    const std::array entries{wgpu::BindGroupEntry{.binding = 0, .buffer = rotation_},
                             wgpu::BindGroupEntry{.binding = 1, .buffer = color}};
    const wgpu::BindGroupDescriptor bindings{
        .layout = pipeline_.GetBindGroupLayout(0), .entryCount = entries.size(), .entries = entries.data(),
    };
    bindings_ = gpu.device().CreateBindGroup(&bindings);
}

void Triangle::encode_draw(const Gpu& gpu, const wgpu::CommandEncoder& encoder,
                      const wgpu::Texture& canvas, float seconds,
                      const wgpu::PassTimestampWrites* timestamps) const
{
    gpu.queue().WriteBuffer(rotation_, 0, &seconds, sizeof(seconds));
    const wgpu::RenderPassColorAttachment color{
        .view = canvas.CreateView(), .loadOp = wgpu::LoadOp::Clear, .storeOp = wgpu::StoreOp::Store,
        .clearValue = {0.012, 0.022, 0.04, 1},
    };
    const wgpu::RenderPassDescriptor descriptor{
        .colorAttachmentCount = 1, .colorAttachments = &color, .timestampWrites = timestamps,
    };
    const auto pass = encoder.BeginRenderPass(&descriptor);
    pass.SetPipeline(pipeline_);
    pass.SetBindGroup(0, bindings_);
    pass.Draw(3);
    pass.End();
}

}
