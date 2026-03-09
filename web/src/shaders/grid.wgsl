struct GridUniform {
  viewProj: mat4x4<f32>,
};

struct VsOut {
  @builtin(position) position: vec4f,
};

@group(0) @binding(0) var<uniform> grid: GridUniform;

@vertex
fn vsMain(@location(0) position: vec3f) -> VsOut {
  var out: VsOut;
  out.position = grid.viewProj * vec4f(position, 1.0);
  return out;
}

@fragment
fn fsMain() -> @location(0) vec4f {
  return vec4f(0.5, 0.82, 1.0, 0.42);
}
