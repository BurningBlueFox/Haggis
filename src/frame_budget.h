#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace haggis {

// All durations are milliseconds. The caller submits at most one chunk at a time.
class FrameBudget {
public:
    static constexpr std::uint32_t max_groups = 2048;
    static constexpr double inference_share = 0.10;

    [[nodiscard]] int target_fps() const { return fps_; }
    [[nodiscard]] double period() const { return 1000.0 / fps_; }

    // Called once per measurement window. A promotion needs two healthy windows.
    void tune(double display_p95, double frame_p95, double render_ms)
    {
        auto sustainable = [&](int fps) {
            const double interval = 1000.0 / fps;
            return display_p95 <= 1.12 * interval && render_ms <= 0.8 * interval;
        };
        // At 144 Hz, drawing at 120 FPS naturally leaves occasional two-refresh gaps.
        const double ticks = std::ceil((period() - 0.25) / std::max(0.1, display_p95));
        const bool late = frame_p95 > 1.12 * std::max(period(), ticks * display_p95);
        lend_ = !late && sustainable(fps_);
        const int ceiling = sustainable(240) ? 240 : sustainable(120) ? 120 : 60;
        if (fps_ > ceiling || late) {
            fps_ = fps_ == 240 ? 120 : 60;
            healthy_windows_ = 0;
        } else if (ceiling > fps_ && ++healthy_windows_ >= 2) {
            fps_ = fps_ == 60 ? 120 : 240;
            healthy_windows_ = 0;
        } else if (ceiling == fps_) {
            healthy_windows_ = 0;
        }
    }

    // Credit accrues only for active work. Bound catch-up after a slow frame.
    void accrue(double elapsed)
    {
        const double earned = inference_share * std::clamp(elapsed, 0.0, 2 * period());
        const double capacity = std::max(0.5 * period(), ms_per_group_);
        credit_ = std::min(capacity, credit_ + earned);
    }

    [[nodiscard]] std::uint32_t take_chunk(double render_ms) const
    {
        const double base = inference_share * period();
        const double slack = std::max(base, 0.9 * period() - render_ms);
        const double allowance = lend_ ? slack : std::min(slack, credit_);
        const auto affordable = std::floor(allowance / ms_per_group_);
        // One indivisible workgroup can cost more than a frame's allowance.
        // Admit it once enough credit has accumulated, even at the lowest tier.
        const double minimum = credit_ >= ms_per_group_ ? 1.0 : 0.0;
        return static_cast<std::uint32_t>(std::clamp(affordable, minimum, double(growth_limit_)));
    }

    void completed(std::uint32_t groups, double elapsed_ms)
    {
        if (groups == 0 || !std::isfinite(elapsed_ms) || elapsed_ms <= 0) {
            return;
        }
        credit_ = std::max(0.0, credit_ - elapsed_ms);
        ms_per_group_ = std::max(0.00001, std::lerp(ms_per_group_, elapsed_ms / groups, 0.30));
        growth_limit_ = std::min(max_groups, groups * 2);
    }

private:
    int fps_ = 120;
    int healthy_windows_ = 0;
    bool lend_ = false;
    double credit_ = 0;
    double ms_per_group_ = 0.08;
    std::uint32_t growth_limit_ = 1;
};

} // namespace haggis
