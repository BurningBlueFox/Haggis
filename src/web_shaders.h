#pragma once

#include <string_view>

namespace haggis {

// Each invocation advances one 4-wide recurrent state through 32 dense layers.
// The state stays in GPU memory between chunks. One lane publishes a display color.
inline constexpr std::string_view compute_shader = R"(
struct Params { time: f32, first_group: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> states: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> color: vec4f;

@compute @workgroup_size(64)
fn compute_main(@builtin(global_invocation_id) id: vec3u) {
    let i = (params.first_group * 64u + id.x) % arrayLength(&states);
    let phase = params.time + f32(i % 256u) * 0.0245;
    let bias = 0.18 * sin(vec4f(phase) + vec4f(0, 2, 4, 6));
    let weights = mat4x4f(vec4f(.76, .21, -.12, .08), vec4f(-.18, .70, .32, .11),
                          vec4f(.10, -.24, .81, .19), vec4f(.26, .09, -.22, .73));
    var x = states[i];
    for (var layer = 0u; layer < 32u; layer++) {
        x = tanh(weights * x + bias);
    }
    states[i] = x;
    if (id.x == 0u) { color = vec4f(0.55 + 0.45 * x.xyz, 1); }
}
)";

inline constexpr std::string_view render_shader = R"(
struct Params { time: f32, first_group: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> color: vec4f;

@vertex fn vertex_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let vertices = array(vec2f(0, .63), vec2f(-.55, -.32), vec2f(.55, -.32));
    let c = cos(params.time);
    let s = sin(params.time);
    let p = mat2x2f(vec2f(c, s), vec2f(-s, c)) * vertices[i];
    return vec4f(p.x * 9.0 / 16.0, p.y, 0, 1);
}
@fragment fn fragment_main() -> @location(0) vec4f { return color; }
)";

} // namespace haggis
