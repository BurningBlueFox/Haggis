#include "frame_budget.h"
#include "web_shaders.h"

#include <emscripten.h>
#include <emscripten/html5.h>
#include <webgpu/webgpu_cpp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <utility>

namespace {

using namespace wgpu;
constexpr auto callback_mode = CallbackMode::AllowSpontaneous;
constexpr std::uint32_t lanes = 64;
constexpr std::size_t query_bytes = 4 * sizeof(std::uint64_t);

EM_JS(void, show_status, (const char* text), {
    document.getElementById('status').textContent = UTF8ToString(text);
});
EM_JS(void, show_metrics, (const char* text), {
    document.getElementById('metrics').textContent = UTF8ToString(text);
});
EM_JS(bool, compute_enabled, (), { return document.getElementById('compute').checked; });

// One second of samples fits even at 240 Hz. Sort only when refreshing the HUD.
struct Samples {
    std::array<double, 512> values{};
    std::size_t count = 0;

    void add(double value) {
        if (count < values.size()) values[count++] = value;
    }

    double take_p95(double fallback) {
        if (!count) return fallback;
        const auto index = (95 * count + 99) / 100 - 1;
        std::nth_element(values.begin(), values.begin() + index, values.begin() + count);
        count = 0;
        return values[index];
    }
};

struct Renderer {
    Instance instance;
    Surface surface;
    Device device;
    Queue queue;
    Buffer params, resolve, readback;
    RenderPipeline render;
    ComputePipeline compute;
    std::array<BindGroup, 2> render_bindings, compute_bindings;
    QuerySet queries;

    haggis::FrameBudget budget;
    Samples raf_intervals, frame_intervals, render_times;
    double previous_tick = 0, previous_draw = 0, next_draw = 0, window_start = 0;
    double render_estimate = 1, submitted_at = 0, compute_time = 0;
    std::uint32_t active_color = 0, cursor = 0, pending_groups = 0;
    std::uint64_t completed_vectors = 0;
    unsigned frame_number = 0;
    bool pending = false, extra_draw = false, stopped = false;

    void fail(const char* reason, StringView detail = {}) {
        if (std::exchange(stopped, true)) return;
        show_status(reason);
        if (detail.data) std::fprintf(stderr, "%.*s\n", int(detail.length), detail.data);
    }

    Buffer buffer(BufferUsage usage, std::uint64_t size) const {
        const BufferDescriptor desc{.usage = usage, .size = size};
        return device.CreateBuffer(&desc);
    }

    ShaderModule shader(std::string_view wgsl) const {
        ShaderSourceWGSL source;
        source.code = wgsl;
        const ShaderModuleDescriptor desc{.nextInChain = &source};
        return device.CreateShaderModule(&desc);
    }

    void start();
    void configure(const Adapter& adapter);
    void create_pipelines(TextureFormat format);
    void draw(double now, std::uint32_t groups, bool sample);
    void finish(double compute_ms);
    bool tick(double now);
};

void Renderer::start()
{
    if (!EM_ASM_INT({ return !!navigator.gpu; })) {
        fail("WebGPU is unavailable in this browser.");
        return;
    }
    instance = CreateInstance();
    EmscriptenSurfaceSourceCanvasHTMLSelector selector;
    selector.selector = "#canvas";
    const SurfaceDescriptor desc{.nextInChain = &selector};
    surface = instance.CreateSurface(&desc);
    instance.RequestAdapter(nullptr, callback_mode,
        [this](RequestAdapterStatus status, Adapter adapter, StringView message) {
            if (status != RequestAdapterStatus::Success) {
                fail("Could not acquire a WebGPU adapter.", message);
                return;
            }
            constexpr auto timestamp = FeatureName::TimestampQuery;
            DeviceDescriptor desc;
            if (adapter.HasFeature(timestamp)) {
                desc.requiredFeatures = &timestamp;
                desc.requiredFeatureCount = 1;
            }
            desc.SetUncapturedErrorCallback(+[](const Device&, ErrorType, StringView msg, Renderer* app) {
                app->fail("WebGPU reported an error. See the browser console.", msg);
            }, this);
            desc.SetDeviceLostCallback(callback_mode, [this](const Device&, DeviceLostReason, StringView msg) {
                fail("WebGPU device lost. Reload to restart.", msg);
            });
            adapter.RequestDevice(&desc, callback_mode,
                [this, adapter](RequestDeviceStatus result, Device acquired, StringView msg) {
                    if (result != RequestDeviceStatus::Success) {
                        fail("Could not create a WebGPU device.", msg);
                        return;
                    }
                    device = std::move(acquired);
                    configure(adapter);
                });
        });
}

void Renderer::configure(const Adapter& adapter)
{
    SurfaceCapabilities caps;
    if (!surface.GetCapabilities(adapter, &caps) || caps.formatCount == 0) {
        fail("No compatible canvas format.");
        return;
    }
    int width{}, height{};
    emscripten_get_canvas_element_size("#canvas", &width, &height);
    const SurfaceConfiguration config{.device = device, .format = caps.formats[0],
        .width = static_cast<unsigned>(width), .height = static_cast<unsigned>(height)};
    surface.Configure(&config);
    queue = device.GetQueue();
    create_pipelines(config.format);
    if (device.HasFeature(FeatureName::TimestampQuery)) {
        const QuerySetDescriptor desc{.type = QueryType::Timestamp, .count = 4};
        queries = device.CreateQuerySet(&desc);
        resolve = buffer(BufferUsage::QueryResolve | BufferUsage::CopySrc, query_bytes);
        readback = buffer(BufferUsage::CopyDst | BufferUsage::MapRead, query_bytes);
    }
    if (stopped) return;
    show_status("WebGPU ready");
    emscripten_request_animation_frame_loop([](double now, void* user) {
        return static_cast<Renderer*>(user)->tick(now);
    }, this);
}

void Renderer::create_pipelines(TextureFormat format)
{
    const auto graphics = shader(haggis::render_shader);
    const ColorTargetState target{.format = format};
    const FragmentState fragment{.module = graphics, .entryPoint = "fragment_main",
                                 .targetCount = 1, .targets = &target};
    const RenderPipelineDescriptor render_desc{
        .vertex = {.module = graphics, .entryPoint = "vertex_main"}, .fragment = &fragment};
    render = device.CreateRenderPipeline(&render_desc);
    const ComputePipelineDescriptor compute_desc{
        .compute = {.module = shader(haggis::compute_shader), .entryPoint = "compute_main"}};
    compute = device.CreateComputePipeline(&compute_desc);

    params = buffer(BufferUsage::Uniform | BufferUsage::CopyDst, 16);
    const auto states = buffer(BufferUsage::Storage, haggis::FrameBudget::max_groups * lanes * 16);
    constexpr std::array initial_color{0.30f, 0.85f, 0.75f, 1.0f};
    for (unsigned i = 0; i < render_bindings.size(); ++i) {
        const auto color = buffer(BufferUsage::Uniform | BufferUsage::Storage | BufferUsage::CopyDst, 16);
        queue.WriteBuffer(color, 0, initial_color.data(), sizeof(initial_color));
        const std::array draw_entries{BindGroupEntry{.binding = 0, .buffer = params},
                                      BindGroupEntry{.binding = 1, .buffer = color}};
        const BindGroupDescriptor draw_desc{.layout = render.GetBindGroupLayout(0),
            .entryCount = draw_entries.size(), .entries = draw_entries.data()};
        render_bindings[i] = device.CreateBindGroup(&draw_desc);

        const std::array work_entries{draw_entries[0], BindGroupEntry{.binding = 1, .buffer = states},
                                      BindGroupEntry{.binding = 2, .buffer = color}};
        const BindGroupDescriptor work_desc{.layout = compute.GetBindGroupLayout(0),
            .entryCount = work_entries.size(), .entries = work_entries.data()};
        compute_bindings[i] = device.CreateBindGroup(&work_desc);
    }
}

void Renderer::draw(double now, std::uint32_t groups, bool sample)
{
    SurfaceTexture frame;
    surface.GetCurrentTexture(&frame);
    if (!frame.texture) {
        fail("Could not acquire the canvas texture. Reload to restart.");
        return;
    }
    struct Params { float time; std::uint32_t first_group; };
    const Params data{static_cast<float>(std::fmod(now / 1000.0, 3600.0)), cursor};
    queue.WriteBuffer(params, 0, &data, sizeof(data));
    const bool timed = sample && queries;
    const PassTimestampWrites render_stamp{.querySet = queries, .beginningOfPassWriteIndex = 0,
                                            .endOfPassWriteIndex = 1};
    const RenderPassColorAttachment color{.view = frame.texture.CreateView(), .loadOp = LoadOp::Clear,
        .storeOp = StoreOp::Store, .clearValue = {0.012, 0.022, 0.04, 1}};
    const RenderPassDescriptor draw_desc{.colorAttachmentCount = 1, .colorAttachments = &color,
                                         .timestampWrites = timed ? &render_stamp : nullptr};
    const auto encoder = device.CreateCommandEncoder();
    const auto pass = encoder.BeginRenderPass(&draw_desc);
    pass.SetPipeline(render);
    pass.SetBindGroup(0, render_bindings[active_color]);
    pass.Draw(3);
    pass.End();

    if (groups) {
        const PassTimestampWrites compute_stamp{.querySet = queries, .beginningOfPassWriteIndex = 2,
                                                .endOfPassWriteIndex = 3};
        const ComputePassDescriptor desc{.timestampWrites = timed ? &compute_stamp : nullptr};
        const auto work = encoder.BeginComputePass(&desc);
        work.SetPipeline(compute);
        work.SetBindGroup(0, compute_bindings[1 - active_color]);
        work.DispatchWorkgroups(groups);
        work.End();
        cursor = (cursor + groups) % haggis::FrameBudget::max_groups;
    }
    const unsigned count = groups ? 4 : 2;
    if (timed) {
        encoder.ResolveQuerySet(queries, 0, count, resolve, 0);
        encoder.CopyBufferToBuffer(resolve, 0, readback, 0, count * sizeof(std::uint64_t));
    }
    const auto commands = encoder.Finish();
    queue.Submit(1, &commands);
    if (!sample) return;
    pending = true;
    pending_groups = groups;
    submitted_at = emscripten_get_now();
    if (timed) {
        readback.MapAsync(MapMode::Read, 0, count * sizeof(std::uint64_t), callback_mode,
            [this, count](MapAsyncStatus status, StringView message) {
                if (status != MapAsyncStatus::Success) {
                    fail("Could not read WebGPU timing data.", message);
                    return;
                }
                const auto* stamps = static_cast<const std::uint64_t*>(
                    readback.GetConstMappedRange(0, count * sizeof(std::uint64_t)));
                if (stamps[1] >= stamps[0]) render_times.add((stamps[1] - stamps[0]) * 1e-6);
                // Equal stamps are common for tiny passes on privacy-coarsened timers.
                const double cost = count == 4 && stamps[3] >= stamps[2]
                    ? (stamps[3] - stamps[2]) * 1e-6 : 0;
                readback.Unmap();
                finish(cost);
            });
    } else {
        queue.OnSubmittedWorkDone(callback_mode, [this](QueueWorkDoneStatus status, StringView msg) {
            if (status != QueueWorkDoneStatus::Success) {
                fail("The WebGPU queue stopped.", msg);
                return;
            }
            const double latency = emscripten_get_now() - submitted_at;
            if (!pending_groups) render_times.add(latency);
            finish(std::max(0.05, latency - render_estimate));
        });
    }
}

void Renderer::finish(double compute_ms)
{
    if (pending_groups) {
        active_color = 1 - active_color;
        completed_vectors += pending_groups * lanes;
        compute_time += compute_ms;
        budget.completed(pending_groups, std::max(0.05, compute_ms));
    }
    pending = false;
    extra_draw = false;
}

bool Renderer::tick(double now)
{
    if (stopped) return false;
    if (!previous_tick || now - previous_tick > 250) {
        window_start = next_draw = now;
        previous_draw = 0;
        raf_intervals.count = frame_intervals.count = render_times.count = 0;
        completed_vectors = 0;
        compute_time = 0;
    } else {
        raf_intervals.add(now - previous_tick);
    }
    previous_tick = now;
    if (now - window_start >= 1000) {
        const double display = raf_intervals.take_p95(budget.period());
        const double frame = frame_intervals.take_p95(now - window_start);
        render_estimate = render_times.take_p95(render_estimate);
        budget.tune(display, frame, render_estimate);
        const double window_ms = now - window_start;
        char metrics[192];
        std::snprintf(metrics, sizeof(metrics),
            "Target %d FPS  ·  Frame p95 %.1f ms  ·  %.0f vectors/s  ·  Compute %.1f%%  ·  %s",
            budget.target_fps(), frame, 1000.0 * completed_vectors / window_ms,
            100 * compute_time / window_ms,
            queries ? "GPU timestamps" : "Queue latency estimate");
        show_metrics(metrics);
        completed_vectors = 0;
        window_start = now;
        compute_time = 0;
    }
    if (now + 0.25 < next_draw || (pending && extra_draw)) return true;

    const double elapsed = previous_draw ? now - previous_draw : budget.period();
    if (previous_draw) frame_intervals.add(elapsed);
    previous_draw = now;
    // Preserve phase on displays whose refresh isn't a multiple of the target.
    next_draw += budget.period();
    if (next_draw <= now) next_draw += std::ceil((now - next_draw) / budget.period()) * budget.period();

    std::uint32_t groups = 0;
    if (compute_enabled()) {
        budget.accrue(elapsed);
        // Periodically isolate rendering cost on adapters without pass timestamps.
        if (!pending && (queries || frame_number % 30 != 0)) groups = budget.take_chunk(render_estimate);
    }
    ++frame_number;
    if (pending) extra_draw = true;
    draw(now, groups, !pending);
    return !stopped;
}

} // namespace

int main()
{
    static Renderer app;
    app.start();
}
