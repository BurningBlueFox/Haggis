#include "inference.h"

#include "frame_budget.h"

#include <array>

namespace haggis::web {
namespace {

struct Parameters {
    float time;
    std::uint32_t first_group;
};

// Each lane advances one four-component state. The first lane also updates the triangle color.
constexpr std::string_view shader = R"(
struct Params { time: f32, first_group: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> states: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> color: vec4f;
override lanes: u32;

@compute @workgroup_size(lanes)
fn compute_main(@builtin(global_invocation_id) id: vec3u) {
    let index = (params.first_group * lanes + id.x) % arrayLength(&states);
    let phase = params.time + f32(index % 256u) * 0.0245;
    let bias = 0.18 * sin(vec4f(phase) + vec4f(0, 2, 4, 6));
    let weights = mat4x4f(vec4f(.76, .21, -.12, .08), vec4f(-.18, .70, .32, .11),
                          vec4f(.10, -.24, .81, .19), vec4f(.26, .09, -.22, .73));
    var state = states[index];
    for (var layer = 0u; layer < 32u; layer++) {
        state = tanh(weights * state + bias);
    }
    states[index] = state;
    if (id.x == 0u) {
        color = vec4f(0.55 + 0.45 * state.xyz, 1);
    }
}
)";

}

void Inference::initialize(const Gpu& gpu)
{
    const wgpu::ConstantEntry lanes{.key = "lanes", .value = vectors_per_group};
    const wgpu::ComputePipelineDescriptor descriptor{
        .compute = {.module = gpu.shader(shader), .entryPoint = "compute_main",
                    .constantCount = 1, .constants = &lanes},
    };
    pipeline_ = gpu.device().CreateComputePipeline(&descriptor);

    constexpr std::array initial_color{0.30f, 0.85f, 0.75f, 1.0f};
    using wgpu::BufferUsage;
    parameters_ = gpu.buffer(BufferUsage::Uniform | BufferUsage::CopyDst, sizeof(Parameters));
    color_ = gpu.buffer(BufferUsage::Uniform | BufferUsage::Storage | BufferUsage::CopyDst, sizeof(initial_color));
    gpu.queue().WriteBuffer(color_, 0, initial_color.data(), sizeof(initial_color));
    const auto states = gpu.buffer(BufferUsage::Storage, FrameBudget::max_groups * vectors_per_group * sizeof(initial_color));

    const std::array entries{wgpu::BindGroupEntry{.binding = 0, .buffer = parameters_},
                             wgpu::BindGroupEntry{.binding = 1, .buffer = states},
                             wgpu::BindGroupEntry{.binding = 2, .buffer = color_}};
    const wgpu::BindGroupDescriptor bindings{
        .layout = pipeline_.GetBindGroupLayout(0), .entryCount = entries.size(), .entries = entries.data(),
    };
    bindings_ = gpu.device().CreateBindGroup(&bindings);
}

void Inference::encode_chunk(const Gpu& gpu, const wgpu::CommandEncoder& encoder, float seconds,
                       std::uint32_t groups, const wgpu::PassTimestampWrites* timestamps)
{
    const Parameters parameters{seconds, first_group_};
    gpu.queue().WriteBuffer(parameters_, 0, &parameters, sizeof(parameters));
    const wgpu::ComputePassDescriptor descriptor{.timestampWrites = timestamps};
    const auto pass = encoder.BeginComputePass(&descriptor);
    pass.SetPipeline(pipeline_);
    pass.SetBindGroup(0, bindings_);
    pass.DispatchWorkgroups(groups);
    pass.End();
    first_group_ = (first_group_ + groups) % FrameBudget::max_groups;
}

}
