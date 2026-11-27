#include "gpu_debug.h"

#include <dawn/webgpu_cpp_print.h>
#include <emscripten.h>

#include <iomanip>
#include <limits>
#include <sstream>
#include <string_view>

namespace haggis::web {
namespace {

constexpr std::string_view not_exposed = "Not exposed";
constexpr int limit_name_width = 48;
constexpr int limit_value_width = 16;

EM_JS(void, set_gpu_debug_text, (const char* text), {
    document.getElementById('gpu-debug').textContent = UTF8ToString(text);
});

std::string_view exposed_text(wgpu::StringView text)
{
    const std::string_view value = text;
    return value.empty() ? not_exposed : value;
}

template<typename Number>
void write_number(std::ostream& out, Number value, Number unavailable)
{
    if (value == unavailable) out << not_exposed;
    else out << value;
}

void write_adapter_info(std::ostream& out, const wgpu::Adapter& adapter)
{
    out << "Adapter information\n";
    wgpu::AdapterInfo info;
    if (!adapter.GetInfo(&info)) {
        out << not_exposed << '\n';
        return;
    }

    out << "Vendor: " << exposed_text(info.vendor)
        << "\nArchitecture: " << exposed_text(info.architecture)
        << "\nDevice: " << exposed_text(info.device)
        << "\nDescription: " << exposed_text(info.description)
        << "\nBackend: " << info.backendType
        << "\nAdapter type: " << info.adapterType
        << "\nVendor ID: ";
    write_number(out, info.vendorID, 0u);
    out << "\nDevice ID: ";
    write_number(out, info.deviceID, 0u);
    out << "\nMinimum subgroup size: ";
    write_number(out, info.subgroupMinSize, 0u);
    out << "\nMaximum subgroup size: ";
    write_number(out, info.subgroupMaxSize, 0u);
    out << '\n';
}

template<typename GpuObject>
void write_features(std::ostream& out, std::string_view heading, const GpuObject& object)
{
    out << '\n' << heading << '\n';
    wgpu::SupportedFeatures features;
    object.GetFeatures(&features);
    if (features.featureCount == 0) {
        out << "None\n";
        return;
    }
    for (std::size_t i = 0; i < features.featureCount; ++i) {
        out << "- " << features.features[i] << '\n';
    }
}

template<typename Number>
void write_limit(std::ostream& out, std::string_view name, Number supported, Number enabled)
{
    constexpr auto undefined = std::numeric_limits<Number>::max();
    out << std::setw(limit_name_width) << name << std::setw(limit_value_width);
    write_number(out, supported, undefined);
    write_number(out, enabled, undefined);
    out << '\n';
}

void write_limits(std::ostream& out, const wgpu::Adapter& adapter, const wgpu::Device& device)
{
    wgpu::Limits supported;
    wgpu::Limits enabled;
    if (!adapter.GetLimits(&supported)) supported = {};
    if (!device.GetLimits(&enabled)) enabled = {};
    out << "\nWebGPU limits\n" << std::left
        << std::setw(limit_name_width) << "Limit"
        << std::setw(limit_value_width) << "Adapter" << "Device\n";

    // Keep field names in sync with the port's wgpu::Limits struct.
#define HAGGIS_WRITE_LIMIT(field) write_limit(out, #field, supported.field, enabled.field)
    HAGGIS_WRITE_LIMIT(maxTextureDimension1D);
    HAGGIS_WRITE_LIMIT(maxTextureDimension2D);
    HAGGIS_WRITE_LIMIT(maxTextureDimension3D);
    HAGGIS_WRITE_LIMIT(maxTextureArrayLayers);
    HAGGIS_WRITE_LIMIT(maxBindGroups);
    HAGGIS_WRITE_LIMIT(maxBindGroupsPlusVertexBuffers);
    HAGGIS_WRITE_LIMIT(maxBindingsPerBindGroup);
    HAGGIS_WRITE_LIMIT(maxDynamicUniformBuffersPerPipelineLayout);
    HAGGIS_WRITE_LIMIT(maxDynamicStorageBuffersPerPipelineLayout);
    HAGGIS_WRITE_LIMIT(maxSampledTexturesPerShaderStage);
    HAGGIS_WRITE_LIMIT(maxSamplersPerShaderStage);
    HAGGIS_WRITE_LIMIT(maxStorageBuffersPerShaderStage);
    HAGGIS_WRITE_LIMIT(maxStorageTexturesPerShaderStage);
    HAGGIS_WRITE_LIMIT(maxUniformBuffersPerShaderStage);
    HAGGIS_WRITE_LIMIT(maxUniformBufferBindingSize);
    HAGGIS_WRITE_LIMIT(maxStorageBufferBindingSize);
    HAGGIS_WRITE_LIMIT(minUniformBufferOffsetAlignment);
    HAGGIS_WRITE_LIMIT(minStorageBufferOffsetAlignment);
    HAGGIS_WRITE_LIMIT(maxVertexBuffers);
    HAGGIS_WRITE_LIMIT(maxBufferSize);
    HAGGIS_WRITE_LIMIT(maxVertexAttributes);
    HAGGIS_WRITE_LIMIT(maxVertexBufferArrayStride);
    HAGGIS_WRITE_LIMIT(maxInterStageShaderVariables);
    HAGGIS_WRITE_LIMIT(maxColorAttachments);
    HAGGIS_WRITE_LIMIT(maxColorAttachmentBytesPerSample);
    HAGGIS_WRITE_LIMIT(maxComputeWorkgroupStorageSize);
    HAGGIS_WRITE_LIMIT(maxComputeInvocationsPerWorkgroup);
    HAGGIS_WRITE_LIMIT(maxComputeWorkgroupSizeX);
    HAGGIS_WRITE_LIMIT(maxComputeWorkgroupSizeY);
    HAGGIS_WRITE_LIMIT(maxComputeWorkgroupSizeZ);
    HAGGIS_WRITE_LIMIT(maxComputeWorkgroupsPerDimension);
    HAGGIS_WRITE_LIMIT(maxImmediateSize);
#undef HAGGIS_WRITE_LIMIT
    out << std::right;
}

}

void show_gpu_debug(const char* text)
{
    set_gpu_debug_text(text);
}

void report_gpu_debug(const wgpu::Adapter& adapter, const wgpu::Device& device)
{
    std::ostringstream text;
    write_adapter_info(text, adapter);
    write_features(text, "Adapter features", adapter);
    write_features(text, "Enabled device features", device);
    write_limits(text, adapter, device);
    show_gpu_debug(text.str().c_str());
}

}
