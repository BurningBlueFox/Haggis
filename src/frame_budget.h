#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace haggis {

// All durations are milliseconds. The caller submits at most one chunk at a time.
class FrameBudget {
public:
    static constexpr int low_fps = 60;
    static constexpr int default_fps = 120;
    static constexpr int high_fps = 240;
    static constexpr double frame_tolerance_ms = 0.25;
    static constexpr double ms_per_second = 1000.0;
    static constexpr double inference_share = 0.10;
    static constexpr std::uint32_t max_groups = 2048;

    [[nodiscard]] int target_fps() const { return fps_; }
    [[nodiscard]] double period() const { return period_for(fps_); }

    void update_target(double display_interval_ms, double frame_p95_ms, double render_ms)
    {
        const int ceiling = sustainable_tier(display_interval_ms, render_ms);
        const bool missed_target = missed_frame_deadline(display_interval_ms, frame_p95_ms);
        lend_ = !missed_target && fits_tier(fps_, display_interval_ms, render_ms);

        if (fps_ > ceiling || missed_target) {
            fps_ = fps_ == high_fps ? default_fps : low_fps;
            healthy_windows_ = 0;
        } else if (fps_ == ceiling) {
            healthy_windows_ = 0;
        } else if (++healthy_windows_ == promotion_windows) {
            fps_ = fps_ == low_fps ? default_fps : high_fps;
            healthy_windows_ = 0;
        }
    }

    // Credit accrues only for active work. Bound catch-up after a slow frame.
    void add_frame_credit(double elapsed_ms)
    {
        const double frame_ms = period();
        const double earned = inference_share * std::clamp(elapsed_ms, 0.0, max_credit_frames * frame_ms);
        const double capacity = std::max(credit_capacity_frames * frame_ms, ms_per_group_);
        credit_ = std::min(capacity, credit_ + earned);
    }

    [[nodiscard]] std::uint32_t workgroups_for(double render_ms) const
    {
        const double frame_ms = period();
        const double reserved_ms = inference_share * frame_ms;
        const double available_ms = std::max(reserved_ms, frame_ms * (1 - headroom_share) - render_ms);
        const double allowance_ms = lend_ ? available_ms : std::min(available_ms, credit_);

        // An indivisible workgroup can exceed a single frame's allowance.
        // Admit it once enough credit has accumulated.
        const double minimum = credit_ >= ms_per_group_ ? 1.0 : 0.0;
        return static_cast<std::uint32_t>(
            std::clamp(allowance_ms / ms_per_group_, minimum, static_cast<double>(growth_limit_)));
    }

    void record_completion(std::uint32_t groups, double elapsed_ms)
    {
        if (groups == 0 || !std::isfinite(elapsed_ms) || elapsed_ms <= 0) return;

        credit_ = std::max(0.0, credit_ - elapsed_ms);
        const double measured_ms = elapsed_ms / groups;
        ms_per_group_ = std::max(min_group_cost_ms, std::lerp(ms_per_group_, measured_ms, sample_weight));
        growth_limit_ = std::min(max_groups, groups * max_chunk_growth);
    }

private:
    static constexpr int promotion_windows = 2;
    static constexpr double timing_margin = 1.12;
    static constexpr double max_render_share = 0.80;
    static constexpr double headroom_share = 0.10;
    static constexpr double min_display_interval_ms = 0.1;
    static constexpr double max_credit_frames = 2;
    static constexpr double credit_capacity_frames = 0.5;
    static constexpr double initial_group_cost_ms = 0.08;
    static constexpr double min_group_cost_ms = 0.00001;
    static constexpr double sample_weight = 0.30;
    static constexpr std::uint32_t max_chunk_growth = 2;

    static constexpr double period_for(int fps) { return ms_per_second / fps; }

    static bool fits_tier(int fps, double display_interval_ms, double render_ms)
    {
        const double frame_ms = period_for(fps);
        return display_interval_ms <= timing_margin * frame_ms && render_ms <= max_render_share * frame_ms;
    }

    static int sustainable_tier(double display_interval_ms, double render_ms)
    {
        if (fits_tier(high_fps, display_interval_ms, render_ms)) return high_fps;
        return fits_tier(default_fps, display_interval_ms, render_ms) ? default_fps : low_fps;
    }

    [[nodiscard]] bool missed_frame_deadline(double display_interval_ms, double frame_p95_ms) const
    {
        const double frame_ms = period();
        // A 120 FPS draw loop on a 144 Hz display can take two refreshes between draws.
        const double refreshes = std::ceil(
            (frame_ms - frame_tolerance_ms) / std::max(min_display_interval_ms, display_interval_ms));
        const double expected_gap_ms = std::max(frame_ms, refreshes * display_interval_ms);
        return frame_p95_ms > timing_margin * expected_gap_ms;
    }

    int fps_ = default_fps;
    int healthy_windows_ = 0;
    bool lend_ = false;
    double credit_ = 0;
    double ms_per_group_ = initial_group_cost_ms;
    std::uint32_t growth_limit_ = 1;
};

} // namespace haggis
