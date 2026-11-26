#pragma once

#include "gpu.h"

#include <cstdint>
#include <optional>

namespace haggis::web {

struct GpuSample {
    std::uint32_t compute_groups;
    std::optional<double> render_ms;
    std::optional<double> compute_ms;
};

// Samples are optional. A busy readback buffer only skips a timing sample.
class GpuTiming {
public:
    void initialize(const Gpu& gpu);

    [[nodiscard]] bool has_timestamps() const { return static_cast<bool>(queries_); }
    [[nodiscard]] bool should_sample(std::uint32_t compute_groups);
    [[nodiscard]] const wgpu::PassTimestampWrites* render_timestamps() const { return &render_stamp_; }
    [[nodiscard]] const wgpu::PassTimestampWrites* compute_timestamps() const { return &compute_stamp_; }

    void encode_readback(const wgpu::CommandEncoder& encoder, std::uint32_t groups) const;
    void read_sample(std::uint32_t groups);
    [[nodiscard]] std::optional<GpuSample> take_sample();

private:
    void complete_read(std::uint32_t groups);

    wgpu::QuerySet queries_;
    wgpu::Buffer resolved_;
    wgpu::Buffer readback_;
    wgpu::PassTimestampWrites render_stamp_{};
    wgpu::PassTimestampWrites compute_stamp_{};
    bool reading_ = false;
    unsigned render_countdown_ = 0;
    unsigned compute_countdown_ = 0;
    std::optional<GpuSample> sample_;
};

}
