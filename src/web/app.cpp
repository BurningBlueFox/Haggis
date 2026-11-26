#include "app.h"

#include "frame_budget.h"
#include "gpu_timing.h"
#include "inference.h"
#include "triangle.h"

#include <emscripten.h>
#include <emscripten/html5.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

namespace haggis::web {
namespace {

constexpr double metrics_interval_ms = FrameBudget::ms_per_second;
constexpr double suspended_after_ms = 250;
constexpr double minimum_measured_cost_ms = 0.05;
constexpr double initial_render_estimate_ms = 1.0;
constexpr double animation_wrap_seconds = 3600.0;
constexpr unsigned render_sample_interval_frames = 30;
constexpr std::size_t metrics_text_capacity = 192;
constexpr std::size_t sample_capacity = 512;
constexpr std::size_t frame_percentile = 95;
constexpr std::size_t percent_scale = 100;

EM_JS(void, show_metrics, (const char* text), {
    document.getElementById('metrics').textContent = UTF8ToString(text);
});

EM_JS(bool, compute_enabled, (), {
    return document.getElementById('compute').checked;
});

class Samples {
public:
    void add(double value)
    {
        if (size_ < values_.size()) values_[size_++] = value;
    }

    double take_p95(double fallback)
    {
        if (!size_) return fallback;
        const auto rank = (frame_percentile * size_ + percent_scale - 1) / percent_scale;
        const auto index = rank - 1;
        std::nth_element(values_.begin(), values_.begin() + index, values_.begin() + size_);
        size_ = 0;
        return values_[index];
    }

    void clear() { size_ = 0; }

private:
    std::array<double, sample_capacity> values_{};
    std::size_t size_ = 0;
};

class App {
public:
    void start();

private:
    void initialize();
    bool animate(double now);
    void observe_refresh(double now);
    double record_draw(double now);
    void submit_frame(double now, std::uint32_t groups);
    std::uint32_t schedule_compute(double elapsed_ms);
    void collect_completed_frame();
    void reset_metrics(double now);
    void report_metrics(double now);

    Gpu gpu_;
    Triangle triangle_;
    Inference inference_;
    GpuTiming timing_;
    FrameBudget budget_;
    Samples display_intervals_, frame_intervals_, render_times_;
    double previous_tick_ = 0;
    double previous_draw_ = 0;
    double next_draw_ = 0;
    double window_start_ = 0;
    double render_ms_ = initial_render_estimate_ms;
    double compute_ms_ = 0;
    std::uint64_t completed_vectors_ = 0;
    unsigned frames_until_render_sample_ = 0;
};

void App::start()
{
    gpu_.connect([this] { initialize(); });
}

void App::initialize()
{
    inference_.initialize(gpu_);
    triangle_.initialize(gpu_, inference_.color());
    timing_.initialize(gpu_);
    if (gpu_.failed()) return;

    std::puts("WebGPU ready");
    emscripten_request_animation_frame_loop([](double now, void* app) {
        return static_cast<App*>(app)->animate(now);
    }, this);
}

bool App::animate(double now)
{
    if (gpu_.failed()) return false;

    observe_refresh(now);
    collect_completed_frame();
    if (now - window_start_ >= metrics_interval_ms) report_metrics(now);

    if (now + FrameBudget::frame_tolerance_ms < next_draw_ || !timing_.can_submit()) return true;

    const double elapsed_ms = record_draw(now);
    const auto groups = schedule_compute(elapsed_ms);
    submit_frame(now, groups);
    return !gpu_.failed();
}

void App::observe_refresh(double now)
{
    const double elapsed_ms = now - previous_tick_;
    if (!previous_tick_ || elapsed_ms > suspended_after_ms) {
        reset_metrics(now);
    } else {
        display_intervals_.add(elapsed_ms);
    }
    previous_tick_ = now;
}

double App::record_draw(double now)
{
    const double period_ms = budget_.period();
    const double elapsed_ms = previous_draw_ ? now - previous_draw_ : period_ms;
    if (previous_draw_) frame_intervals_.add(elapsed_ms);
    previous_draw_ = now;

    // Advance by one period normally. Skip missed slots without shifting the cadence.
    next_draw_ += period_ms;
    if (next_draw_ <= now) {
        const double missed = std::floor((now - next_draw_) / period_ms) + 1;
        next_draw_ += missed * period_ms;
    }
    return elapsed_ms;
}

std::uint32_t App::schedule_compute(double elapsed_ms)
{
    if (!compute_enabled()) return 0;
    budget_.add_frame_credit(elapsed_ms);
    if (timing_.busy()) return 0;

    // A render-only sample isolates render time when only queue timing is available.
    if (!timing_.has_timestamps() && frames_until_render_sample_ == 0) {
        frames_until_render_sample_ = render_sample_interval_frames;
        return 0;
    }
    if (frames_until_render_sample_) --frames_until_render_sample_;
    return budget_.workgroups_for(render_ms_);
}

void App::submit_frame(double now, std::uint32_t groups)
{
    const auto texture = gpu_.canvas_texture();
    if (!texture) {
        gpu_.fail("Could not acquire the canvas texture. Reload to restart.");
        return;
    }

    const auto seconds = static_cast<float>(std::fmod(now / FrameBudget::ms_per_second, animation_wrap_seconds));
    const auto encoder = gpu_.device().CreateCommandEncoder();
    triangle_.encode_draw(gpu_, encoder, texture, seconds, timing_.render_timestamps());
    // Queue order lets the draw read the previous color before compute updates it.
    if (groups) inference_.encode_chunk(gpu_, encoder, seconds, groups, timing_.compute_timestamps());
    timing_.submit(gpu_, encoder, groups);
}

void App::collect_completed_frame()
{
    const auto frame = timing_.take_completed();
    if (!frame) return;

    if (frame->render_ms) {
        render_times_.add(*frame->render_ms);
    } else if (!timing_.has_timestamps() && frame->compute_groups == 0) {
        render_times_.add(frame->queue_ms);
    }
    if (frame->compute_groups == 0) return;

    const double cost = frame->compute_ms.value_or(std::max(0.0, frame->queue_ms - render_ms_));
    compute_ms_ += cost;
    budget_.record_completion(frame->compute_groups, std::max(minimum_measured_cost_ms, cost));
    completed_vectors_ += frame->compute_groups * Inference::vectors_per_group;
}

void App::reset_metrics(double now)
{
    window_start_ = next_draw_ = now;
    previous_draw_ = 0;
    display_intervals_.clear();
    frame_intervals_.clear();
    render_times_.clear();
    completed_vectors_ = 0;
    compute_ms_ = 0;
}

void App::report_metrics(double now)
{
    const double window_ms = now - window_start_;
    const double display = display_intervals_.take_p95(budget_.period());
    const double frame = frame_intervals_.take_p95(window_ms);
    render_ms_ = render_times_.take_p95(render_ms_);
    budget_.update_target(display, frame, render_ms_);

    char metrics[metrics_text_capacity];
    std::snprintf(metrics, sizeof(metrics),
        "Target %d FPS  ·  Frame p95 %.1f ms  ·  %.0f vectors/s  ·  Compute %.1f%%  ·  %s",
        budget_.target_fps(), frame, FrameBudget::ms_per_second * completed_vectors_ / window_ms,
        percent_scale * compute_ms_ / window_ms, timing_.has_timestamps() ? "GPU timestamps" : "Queue latency estimate");
    show_metrics(metrics);
    completed_vectors_ = 0;
    compute_ms_ = 0;
    window_start_ = now;
}

}

void run()
{
    static App app;
    app.start();
}

}
