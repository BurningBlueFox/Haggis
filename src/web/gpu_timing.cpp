#include "gpu_timing.h"

#include <emscripten.h>

#include <array>
#include <cstring>
#include <utility>

namespace haggis::web {
namespace {

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

void GpuTiming::submit(Gpu& gpu, const wgpu::CommandEncoder& encoder, std::uint32_t compute_groups)
{
    if (!waiting_ && queries_) {
        encode_timestamp_copy(encoder, compute_groups);
    }
    const auto commands = encoder.Finish();
    gpu.queue().Submit(1, &commands);

    if (waiting_) {
        draw_queued_ = true;
        return;
    }
    waiting_ = true;
    if (queries_) {
        read_timestamps(gpu, compute_groups);
    } else {
        wait_for_queue(gpu, compute_groups);
    }
}

void GpuTiming::encode_timestamp_copy(const wgpu::CommandEncoder& encoder, std::uint32_t groups) const
{
    const auto count = timestamp_count(groups);
    encoder.ResolveQuerySet(queries_, render_start, count, resolved_, 0);
    encoder.CopyBufferToBuffer(resolved_, 0, readback_, 0, timestamp_bytes(count));
}

void GpuTiming::read_timestamps(Gpu& gpu, std::uint32_t groups)
{
    const auto bytes = timestamp_bytes(timestamp_count(groups));
    readback_.MapAsync(wgpu::MapMode::Read, 0, bytes, asynchronous,
        [this, &gpu, groups](wgpu::MapAsyncStatus status, wgpu::StringView detail) {
            if (status != wgpu::MapAsyncStatus::Success) {
                gpu.fail("Could not read WebGPU timing data.", detail);
                return;
            }

            complete_timestamp_read(groups);
        });
}

void GpuTiming::complete_timestamp_read(std::uint32_t groups)
{
    TimestampValues stamps{};
    const auto bytes = timestamp_bytes(timestamp_count(groups));
    std::memcpy(stamps.data(), readback_.GetConstMappedRange(0, bytes), bytes);
    readback_.Unmap();
    complete({
        .compute_groups = groups,
        .render_ms = elapsed_ms(stamps[render_start], stamps[render_end]),
        .compute_ms = groups ? elapsed_ms(stamps[compute_start], stamps[compute_end]) : std::nullopt,
    });
}

void GpuTiming::wait_for_queue(Gpu& gpu, std::uint32_t groups)
{
    const double submitted_at = emscripten_get_now();
    gpu.queue().OnSubmittedWorkDone(asynchronous,
        [this, &gpu, groups, submitted_at](wgpu::QueueWorkDoneStatus status, wgpu::StringView detail) {
            if (status != wgpu::QueueWorkDoneStatus::Success) {
                gpu.fail("The WebGPU queue stopped.", detail);
                return;
            }
            complete({.compute_groups = groups, .queue_ms = emscripten_get_now() - submitted_at});
        });
}

void GpuTiming::complete(CompletedFrame frame)
{
    completed_ = frame;
    waiting_ = false;
    draw_queued_ = false;
}

std::optional<CompletedFrame> GpuTiming::take_completed()
{
    return std::exchange(completed_, std::nullopt);
}

}
