#include "gpu_timing.h"

#include <emscripten.h>

#include <array>
#include <cstring>
#include <utility>

namespace haggis::web {
namespace {

constexpr unsigned timestamp_count(std::uint32_t groups) { return groups ? 4 : 2; }

std::optional<double> elapsed_ms(std::uint64_t start, std::uint64_t end)
{
    if (end < start) return std::nullopt;
    return (end - start) * 1e-6;
}

}

void GpuTiming::initialize(const Gpu& gpu)
{
    if (!gpu.device().HasFeature(wgpu::FeatureName::TimestampQuery)) return;

    const wgpu::QuerySetDescriptor descriptor{.type = wgpu::QueryType::Timestamp, .count = 4};
    queries_ = gpu.device().CreateQuerySet(&descriptor);
    constexpr auto bytes = 4 * sizeof(std::uint64_t);
    using wgpu::BufferUsage;
    resolved_ = gpu.buffer(BufferUsage::QueryResolve | BufferUsage::CopySrc, bytes);
    readback_ = gpu.buffer(BufferUsage::CopyDst | BufferUsage::MapRead, bytes);
    render_stamp_ = {.querySet = queries_, .beginningOfPassWriteIndex = 0, .endOfPassWriteIndex = 1};
    compute_stamp_ = {.querySet = queries_, .beginningOfPassWriteIndex = 2, .endOfPassWriteIndex = 3};
}

const wgpu::PassTimestampWrites* GpuTiming::render_timestamps() const
{
    return !waiting_ && queries_ ? &render_stamp_ : nullptr;
}

const wgpu::PassTimestampWrites* GpuTiming::compute_timestamps() const
{
    return !waiting_ && queries_ ? &compute_stamp_ : nullptr;
}

void GpuTiming::submit(Gpu& gpu, const wgpu::CommandEncoder& encoder, std::uint32_t compute_groups)
{
    if (!waiting_ && queries_) {
        const auto count = timestamp_count(compute_groups);
        encoder.ResolveQuerySet(queries_, 0, count, resolved_, 0);
        encoder.CopyBufferToBuffer(resolved_, 0, readback_, 0, count * sizeof(std::uint64_t));
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

void GpuTiming::read_timestamps(Gpu& gpu, std::uint32_t groups)
{
    const auto bytes = timestamp_count(groups) * sizeof(std::uint64_t);
    readback_.MapAsync(wgpu::MapMode::Read, 0, bytes, asynchronous,
        [this, &gpu, groups, bytes](wgpu::MapAsyncStatus status, wgpu::StringView detail) {
            if (status != wgpu::MapAsyncStatus::Success) {
                gpu.fail("Could not read WebGPU timing data.", detail);
                return;
            }

            std::array<std::uint64_t, 4> stamps{};
            std::memcpy(stamps.data(), readback_.GetConstMappedRange(0, bytes), bytes);
            readback_.Unmap();
            complete({.compute_groups = groups, .render_ms = elapsed_ms(stamps[0], stamps[1]),
                      .compute_ms = groups ? elapsed_ms(stamps[2], stamps[3]) : std::nullopt});
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
