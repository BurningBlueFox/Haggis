#include "inference.h"

#include "frame_budget.h"

#include <array>
#include <bit>

namespace haggis::web {
namespace {

constexpr std::uint32_t resources_group = 0;
constexpr std::uint32_t parameters_binding = 0;
constexpr std::uint32_t states_binding = 1;
constexpr std::uint32_t color_binding = 2;

using StateVector = std::array<float, 4>;
constexpr StateVector initial_color{0.30f, 0.85f, 0.75f, 1.0f};
constexpr auto state_count = FrameBudget::max_groups * Inference::vectors_per_group;
static_assert(std::has_single_bit(state_count), "The state index mask requires a power-of-two state count");
constexpr auto state_index_mask = state_count - 1;

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
override state_index_mask: u32;

const MODEL_LAYERS = 32u;
const PHASE_COUNT = 256u;
const PHASE_STEP = 0.0245;
const BIAS_SCALE = 0.18;
const BIAS_PHASES = vec4f(0, 2, 4, 6);
const COLOR_BASE = 0.55;
const COLOR_SCALE = 0.45;
const MODEL_WEIGHTS = mat4x4f(
    vec4f(.76, .21, -.12, .08), vec4f(-.18, .70, .32, .11),
    vec4f(.10, -.24, .81, .19), vec4f(.26, .09, -.22, .73));

@compute @workgroup_size(lanes)
fn compute_main(@builtin(global_invocation_id) id: vec3u) {
    let index = (params.first_group * lanes + id.x) & state_index_mask;
    let phase = params.time + f32(index % PHASE_COUNT) * PHASE_STEP;
    let bias = BIAS_SCALE * sin(vec4f(phase) + BIAS_PHASES);
    var state = states[index];
    for (var layer = 0u; layer < MODEL_LAYERS; layer++) {
        state = tanh(MODEL_WEIGHTS * state + bias);
    }
    states[index] = state;
    if (id.x == 0u) {
        color = vec4f(COLOR_BASE + COLOR_SCALE * state.xyz, 1);
    }
}
)";

wgpu::ComputePipeline create_inference_pipeline(const Gpu& gpu)
{
    const std::array constants{
        wgpu::ConstantEntry{.key = "lanes", .value = Inference::vectors_per_group},
        wgpu::ConstantEntry{.key = "state_index_mask", .value = state_index_mask},
    };
    const wgpu::ComputePipelineDescriptor descriptor{
        .compute = {.module = gpu.shader(shader), .entryPoint = "compute_main",
                    .constantCount = constants.size(), .constants = constants.data()},
    };
    return gpu.device().CreateComputePipeline(&descriptor);
}

}

void Inference::initialize(const Gpu& gpu)
{
    pipeline_ = create_inference_pipeline(gpu);
    using wgpu::BufferUsage;
    parameters_ = gpu.buffer(BufferUsage::Uniform | BufferUsage::CopyDst, sizeof(Parameters));
    color_ = gpu.buffer(BufferUsage::Uniform | BufferUsage::Storage | BufferUsage::CopyDst, sizeof(initial_color));
    gpu.queue().WriteBuffer(color_, 0, initial_color.data(), sizeof(initial_color));
    const auto states = gpu.buffer(BufferUsage::Storage, state_count * sizeof(StateVector));

    const std::array entries{wgpu::BindGroupEntry{.binding = parameters_binding, .buffer = parameters_},
                             wgpu::BindGroupEntry{.binding = states_binding, .buffer = states},
                             wgpu::BindGroupEntry{.binding = color_binding, .buffer = color_}};
    const wgpu::BindGroupDescriptor bindings{
        .layout = pipeline_.GetBindGroupLayout(resources_group), .entryCount = entries.size(), .entries = entries.data(),
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
    pass.SetBindGroup(resources_group, bindings_);
    pass.DispatchWorkgroups(groups);
    pass.End();
    first_group_ = (first_group_ + groups) % FrameBudget::max_groups;
}

}
