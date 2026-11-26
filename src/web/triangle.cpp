#include "triangle.h"

#include <array>

namespace haggis::web {
namespace {

constexpr std::uint32_t vertex_count = 3;
constexpr std::uint32_t resources_group = 0;
constexpr std::uint32_t rotation_binding = 0;
constexpr std::uint32_t color_binding = 1;
constexpr wgpu::Color background_color{0.012, 0.022, 0.04, 1.0};

constexpr std::string_view shader = R"(
@group(0) @binding(0) var<uniform> time: f32;
@group(0) @binding(1) var<uniform> color: vec4f;

const VERTICES = array(vec2f(0, .63), vec2f(-.55, -.32), vec2f(.55, -.32));
const VIEWPORT_HEIGHT_OVER_WIDTH: f32 = 9.0 / 16.0;

@vertex fn vertex_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let c = cos(time);
    let s = sin(time);
    let p = mat2x2f(vec2f(c, s), vec2f(-s, c)) * VERTICES[i];
    return vec4f(p.x * VIEWPORT_HEIGHT_OVER_WIDTH, p.y, 0, 1);
}
@fragment fn fragment_main() -> @location(0) vec4f {
    return color;
}
)";

wgpu::RenderPipeline create_triangle_pipeline(const Gpu& gpu)
{
    const auto module = gpu.shader(shader);
    const wgpu::ColorTargetState target{.format = gpu.canvas_format()};
    const wgpu::FragmentState fragment{
        .module = module, .entryPoint = "fragment_main", .targetCount = 1, .targets = &target,
    };
    const wgpu::RenderPipelineDescriptor descriptor{
        .vertex = {.module = module, .entryPoint = "vertex_main"}, .fragment = &fragment,
    };
    return gpu.device().CreateRenderPipeline(&descriptor);
}

}

void Triangle::initialize(const Gpu& gpu, const wgpu::Buffer& color)
{
    pipeline_ = create_triangle_pipeline(gpu);

    rotation_ = gpu.buffer(wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, sizeof(float));
    const std::array entries{wgpu::BindGroupEntry{.binding = rotation_binding, .buffer = rotation_},
                             wgpu::BindGroupEntry{.binding = color_binding, .buffer = color}};
    const wgpu::BindGroupDescriptor bindings{
        .layout = pipeline_.GetBindGroupLayout(resources_group), .entryCount = entries.size(), .entries = entries.data(),
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
        .clearValue = background_color,
    };
    const wgpu::RenderPassDescriptor descriptor{
        .colorAttachmentCount = 1, .colorAttachments = &color, .timestampWrites = timestamps,
    };
    const auto pass = encoder.BeginRenderPass(&descriptor);
    pass.SetPipeline(pipeline_);
    pass.SetBindGroup(resources_group, bindings_);
    pass.Draw(vertex_count);
    pass.End();
}

}
