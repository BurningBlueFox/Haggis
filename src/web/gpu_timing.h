#pragma once

#include "gpu.h"

#include <cstdint>
#include <optional>

namespace haggis::web {

struct CompletedFrame {
    std::uint32_t compute_groups;
    std::optional<double> render_ms;
    std::optional<double> compute_ms;
    double queue_ms = 0;
};

// One measured submission can be outstanding. Allow one more draw to queue behind it.
class GpuTiming {
public:
    void initialize(const Gpu& gpu);
    void submit(Gpu& gpu, const wgpu::CommandEncoder& encoder, std::uint32_t compute_groups);

    [[nodiscard]] bool busy() const { return waiting_; }
    [[nodiscard]] bool can_submit() const { return !waiting_ || !draw_queued_; }
    [[nodiscard]] bool has_timestamps() const { return static_cast<bool>(queries_); }
    [[nodiscard]] const wgpu::PassTimestampWrites* render_timestamps() const;
    [[nodiscard]] const wgpu::PassTimestampWrites* compute_timestamps() const;
    [[nodiscard]] std::optional<CompletedFrame> take_completed();

private:
    void read_timestamps(Gpu& gpu, std::uint32_t groups);
    void wait_for_queue(Gpu& gpu, std::uint32_t groups);
    void complete(CompletedFrame frame);

    wgpu::QuerySet queries_;
    wgpu::Buffer resolved_;
    wgpu::Buffer readback_;
    wgpu::PassTimestampWrites render_stamp_{};
    wgpu::PassTimestampWrites compute_stamp_{};
    bool waiting_ = false;
    bool draw_queued_ = false;
    std::optional<CompletedFrame> completed_;
};

}
