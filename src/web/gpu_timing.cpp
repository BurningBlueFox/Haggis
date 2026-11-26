#include "gpu_timing.h"

#include <array>
#include <cstring>
#include <utility>

namespace haggis::web {
namespace {

constexpr unsigned render_sample_interval = 30;
constexpr unsigned compute_sample_interval = 8;

enum TimestampSlot : std::uint32_t {
    render_start,
    render_end,
    compute_start,
    compute_end,
    query_count,
};
constexpr std::uint32_t render_query_count = render_end + 1;
constexpr double ms_per_ns = 1.0 / 1'000'000.0;
using TimestampValues = std::array<std::uint64_t, query_count>;

constexpr std::uint32_t timestamp_count(std::uint32_t groups)
{
    return groups ? static_cast<std::uint32_t>(query_count) : render_query_count;
}

constexpr std::size_t timestamp_bytes(std::uint32_t count)
{
    return count * sizeof(TimestampValues::value_type);
}

std::optional<double> elapsed_ms(std::uint64_t start, std::uint64_t end)
{
    if (end < start) return std::nullopt;
    return (end - start) * ms_per_ns;
}

}

void GpuTiming::initialize(const Gpu& gpu)
{
    if (!gpu.device().HasFeature(wgpu::FeatureName::TimestampQuery)) return;

    const wgpu::QuerySetDescriptor descriptor{.type = wgpu::QueryType::Timestamp, .count = query_count};
    queries_ = gpu.device().CreateQuerySet(&descriptor);
    constexpr auto bytes = timestamp_bytes(query_count);
    using wgpu::BufferUsage;
    resolved_ = gpu.buffer(BufferUsage::QueryResolve | BufferUsage::CopySrc, bytes);
    readback_ = gpu.buffer(BufferUsage::CopyDst | BufferUsage::MapRead, bytes);
    render_stamp_ = {.querySet = queries_, .beginningOfPassWriteIndex = render_start, .endOfPassWriteIndex = render_end};
    compute_stamp_ = {.querySet = queries_, .beginningOfPassWriteIndex = compute_start, .endOfPassWriteIndex = compute_end};
}

bool GpuTiming::should_sample(std::uint32_t groups)
{
    if (!has_timestamps() || reading_) return false;

    auto& countdown = groups ? compute_countdown_ : render_countdown_;
    if (countdown) {
        --countdown;
        return false;
    }
    const auto interval = groups ? compute_sample_interval : render_sample_interval;
    countdown = interval - 1;
    return true;
}

void GpuTiming::encode_readback(const wgpu::CommandEncoder& encoder, std::uint32_t groups) const
{
    const auto count = timestamp_count(groups);
    encoder.ResolveQuerySet(queries_, render_start, count, resolved_, 0);
    encoder.CopyBufferToBuffer(resolved_, 0, readback_, 0, timestamp_bytes(count));
}

void GpuTiming::read_sample(std::uint32_t groups)
{
    reading_ = true;
    const auto bytes = timestamp_bytes(timestamp_count(groups));
    readback_.MapAsync(wgpu::MapMode::Read, 0, bytes, asynchronous,
        [this, groups](wgpu::MapAsyncStatus status, wgpu::StringView) {
            if (status == wgpu::MapAsyncStatus::Success) {
                complete_read(groups);
            } else {
                queries_ = nullptr;
            }
            reading_ = false;
        });
}

void GpuTiming::complete_read(std::uint32_t groups)
{
    TimestampValues stamps{};
    const auto bytes = timestamp_bytes(timestamp_count(groups));
    std::memcpy(stamps.data(), readback_.GetConstMappedRange(0, bytes), bytes);
    readback_.Unmap();
    sample_ = GpuSample{
        .compute_groups = groups,
        .render_ms = elapsed_ms(stamps[render_start], stamps[render_end]),
        .compute_ms = groups ? elapsed_ms(stamps[compute_start], stamps[compute_end]) : std::nullopt,
    };
}

std::optional<GpuSample> GpuTiming::take_sample()
{
    return std::exchange(sample_, std::nullopt);
}

}
