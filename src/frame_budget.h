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

    // Check the current tier once per measurement window. Require two healthy
    // windows before stepping up to avoid bouncing between tiers.
    void update_target(double display_interval_ms, double frame_p95_ms, double render_ms)
    {
        const auto supports = [&](int fps) {
            const double frame_ms = 1000.0 / fps;
            return display_interval_ms <= 1.12 * frame_ms && render_ms <= 0.8 * frame_ms;
        };
        // A 120 FPS draw loop on a 144 Hz display can take two refreshes between draws.
        const double refreshes = std::ceil((period() - 0.25) / std::max(0.1, display_interval_ms));
        const double expected_gap = std::max(period(), refreshes * display_interval_ms);
        const bool missed_target = frame_p95_ms > 1.12 * expected_gap;
        const int ceiling = supports(240) ? 240 : (supports(120) ? 120 : 60);
        lend_ = !missed_target && supports(fps_);

        if (fps_ > ceiling || missed_target) {
            fps_ = fps_ == 240 ? 120 : 60;
            healthy_windows_ = 0;
            return;
        }

        if (fps_ == ceiling) {
            healthy_windows_ = 0;
            return;
        }

        if (++healthy_windows_ == 2) {
            fps_ = fps_ == 60 ? 120 : 240;
            healthy_windows_ = 0;
        }
    }

    // Credit accrues only for active work. Bound catch-up after a slow frame.
    void add_frame_credit(double elapsed)
    {
        const double earned = inference_share * std::clamp(elapsed, 0.0, 2 * period());
        const double capacity = std::max(0.5 * period(), ms_per_group_);
        credit_ = std::min(capacity, credit_ + earned);
    }

    [[nodiscard]] std::uint32_t workgroups_for(double render_ms) const
    {
        const double reserved_ms = inference_share * period();
        const double slack = std::max(reserved_ms, period() - reserved_ms - render_ms);
        const double allowance = lend_ ? slack : std::min(slack, credit_);
        const auto affordable = std::floor(allowance / ms_per_group_);
        // One indivisible workgroup can cost more than a frame's allowance.
        // Admit it once enough credit has accumulated, even at the lowest tier.
        const double minimum = credit_ >= ms_per_group_ ? 1.0 : 0.0;
        return static_cast<std::uint32_t>(std::clamp(affordable, minimum, double(growth_limit_)));
    }

    void record_completion(std::uint32_t groups, double elapsed_ms)
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
