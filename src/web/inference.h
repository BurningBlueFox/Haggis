#pragma once

#include "gpu.h"

#include <cstdint>

namespace haggis::web {

class Inference {
public:
    static constexpr std::uint32_t vectors_per_group = 64;

    void initialize(const Gpu& gpu);
    void encode_chunk(const Gpu& gpu, const wgpu::CommandEncoder& encoder, float seconds,
                std::uint32_t groups, const wgpu::PassTimestampWrites* timestamps);

    [[nodiscard]] const wgpu::Buffer& color() const { return color_; }

private:
    wgpu::Buffer parameters_;
    wgpu::Buffer color_;
    wgpu::ComputePipeline pipeline_;
    wgpu::BindGroup bindings_;
    std::uint32_t first_group_ = 0;
};

}
