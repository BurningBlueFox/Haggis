#pragma once

#include "gpu.h"

namespace haggis::web {

class Triangle {
public:
    void initialize(const Gpu& gpu, const wgpu::Buffer& color);
    void encode_draw(const Gpu& gpu, const wgpu::CommandEncoder& encoder, const wgpu::Texture& canvas,
                float seconds, const wgpu::PassTimestampWrites* timestamps) const;

private:
    wgpu::Buffer rotation_;
    wgpu::RenderPipeline pipeline_;
    wgpu::BindGroup bindings_;
};

}
