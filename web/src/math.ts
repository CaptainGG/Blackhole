import type { Vec3 } from "./types";

export function vec3(x = 0, y = 0, z = 0): Vec3 {
  return { x, y, z };
}

export function cloneVec3(value: Vec3): Vec3 {
  return { x: value.x, y: value.y, z: value.z };
}

export function add(a: Vec3, b: Vec3): Vec3 {
  return vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

export function subtract(a: Vec3, b: Vec3): Vec3 {
  return vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

export function scale(value: Vec3, factor: number): Vec3 {
  return vec3(value.x * factor, value.y * factor, value.z * factor);
}

export function dot(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

export function cross(a: Vec3, b: Vec3): Vec3 {
  return vec3(
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
  );
}

export function length(value: Vec3): number {
  return Math.hypot(value.x, value.y, value.z);
}

export function normalize(value: Vec3): Vec3 {
  const len = length(value);
  return len > 0 ? scale(value, 1 / len) : vec3();
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

export function perspectiveMatrix(
  fovRadians: number,
  aspect: number,
  near: number,
  far: number
): Float32Array {
  const f = 1.0 / Math.tan(fovRadians / 2.0);
  const nf = 1.0 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, (2 * far * near) * nf, 0
  ]);
}

export function lookAtMatrix(eye: Vec3, target: Vec3, up: Vec3): Float32Array {
  const z = normalize(subtract(eye, target));
  const x = normalize(cross(up, z));
  const y = cross(z, x);

  return new Float32Array([
    x.x, y.x, z.x, 0,
    x.y, y.y, z.y, 0,
    x.z, y.z, z.z, 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1
  ]);
}

export function multiplyMatrices(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let row = 0; row < 4; row += 1) {
    for (let col = 0; col < 4; col += 1) {
      let sum = 0;
      for (let i = 0; i < 4; i += 1) {
        sum += a[row * 4 + i] * b[i * 4 + col];
      }
      out[row * 4 + col] = sum;
    }
  }
  return out;
}
