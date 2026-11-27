#pragma once

#include <webgpu/webgpu_cpp.h>

namespace haggis::web {

void show_gpu_debug(const char* text);
void report_gpu_debug(const wgpu::Adapter& adapter, const wgpu::Device& device);

}
