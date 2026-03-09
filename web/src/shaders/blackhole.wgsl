struct CameraUniform {
  pos: vec4f,
  right: vec4f,
  up: vec4f,
  forward: vec4f,
  params: vec4f,
};

struct SceneUniform {
  disk: vec4f,
  tuning: vec4f,
  flags: vec4f,
};

struct ObjectData {
  posRadius: vec4f,
  color: vec4f,
  velocityMass: vec4f,
};

struct Ray {
  x: f32,
  y: f32,
  z: f32,
  r: f32,
  theta: f32,
  phi: f32,
  dr: f32,
  dtheta: f32,
  dphi: f32,
  energy: f32,
  angularMomentum: f32,
};

struct ObjectHit {
  hit: bool,
  color: vec4f,
  center: vec3f,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> scene: SceneUniform;
@group(0) @binding(2) var<storage, read> objects: array<ObjectData, 16>;
@group(0) @binding(3) var outImage: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var historyImage: texture_2d<f32>;

const D_LAMBDA: f32 = 2.5e8;
const ESCAPE_R: f32 = 1.0e26;

fn hash3(value: vec3u) -> u32 {
  var x = value;
  x = x ^ (x >> vec3u(16u));
  x = x * vec3u(0x7feb352du);
  x = x ^ (x >> vec3u(15u));
  x = x * vec3u(0x846ca68bu);
  x = x ^ (x >> vec3u(16u));
  return x.x ^ x.y ^ x.z;
}

fn hash1(value: vec3f) -> f32 {
  return f32(hash3(vec3u(bitcast<u32>(value.x), bitcast<u32>(value.y), bitcast<u32>(value.z)))) / 4294967295.0;
}

fn safeNormalize(value: vec3f) -> vec3f {
  let len = length(value);
  if (len < 1e-6) {
    return vec3f(0.0, 0.0, 0.0);
  }
  return value / len;
}

fn starfield(dir: vec3f, layerCount: u32) -> vec3f {
  let nDir = safeNormalize(dir);
  var color = vec3f(0.0, 0.0, 0.0);
  for (var layer = 0u; layer < layerCount; layer = layer + 1u) {
    let layerF = f32(layer);
    let g = 36.0 + 12.0 * layerF;
    let sph = vec2f(atan2(nDir.z, nDir.x), acos(clamp(nDir.y, -1.0, 1.0)));
    let cell = floor(sph * vec2f(g, g * 0.5));
    let n = hash1(vec3f(cell, 11.0 + 7.0 * layerF));
    if (n < 0.03) {
      let f = fract(sph * vec2f(g, g * 0.5)) - 0.5;
      let d = length(f);
      let size = mix(0.0010, 0.0032, hash1(vec3f(cell, 5.0 + layerF)));
      let twinkle = 0.8 + 0.2 * sin(camera.params.z * (2.0 + 0.7 * layerF) + n * 20.0);
      let intensity = exp(-(d * d) / (size * size)) * mix(0.7, 1.5, hash1(vec3f(cell, 3.0 + layerF))) * twinkle;
      let tint = mix(vec3f(0.85, 0.9, 1.0), vec3f(1.0, 0.95, 0.8), hash1(vec3f(cell, 9.0 + layerF)));
      color += tint * intensity;
    }
  }
  return color;
}

fn heatRamp(t: f32) -> vec3f {
  let c1 = vec3f(0.95, 0.35, 0.07);
  let c2 = vec3f(1.0, 0.85, 0.25);
  let c3 = vec3f(1.0, 1.0, 1.0);
  let a = mix(c1, c2, smoothstep(0.0, 0.7, clamp(t, 0.0, 1.0)));
  return mix(a, c3, smoothstep(0.6, 1.0, clamp(t, 0.0, 1.0)));
}

fn gravRedshift(r: f32, rs: f32) -> f32 {
  return sqrt(clamp(1.0 - (rs / max(r, rs + 1e-4)), 0.0, 1.0));
}

fn dopplerFactor(beta: f32, mu: f32) -> f32 {
  let numerator = max(1e-6, 1.0 + beta * mu);
  let denominator = max(1e-6, 1.0 - beta * mu);
  return sqrt(numerator / denominator);
}

fn keplerBeta(m: f32, r: f32) -> f32 {
  let velocity = sqrt(max(0.0, m / max(r, 1.0)));
  return clamp(velocity * 1e-19, 0.0, 0.6);
}

fn initRay(position: vec3f, dir: vec3f) -> Ray {
  let r = max(length(position), scene.disk.z * 2.0);
  let theta = acos(clamp(position.z / r, -1.0, 1.0));
  let phi = atan2(position.y, position.x);
  let dx = dir.x;
  let dy = dir.y;
  let dz = dir.z;
  let sinTheta = max(sin(theta), 1e-5);
  let dr = sin(theta) * cos(phi) * dx + sin(theta) * sin(phi) * dy + cos(theta) * dz;
  let dtheta = (cos(theta) * cos(phi) * dx + cos(theta) * sin(phi) * dy - sin(theta) * dz) / r;
  let dphi = (-sin(phi) * dx + cos(phi) * dy) / (r * sinTheta);
  let f = 1.0 - scene.disk.z / r;
  let dtdL = sqrt((dr * dr) / max(f, 1e-4) + r * r * (dtheta * dtheta + sinTheta * sinTheta * dphi * dphi));
  return Ray(position.x, position.y, position.z, r, theta, phi, dr, dtheta, dphi, f * dtdL, r * r * sinTheta * dphi);
}

fn geodesicStep(ray: ptr<function, Ray>, dL: f32) {
  let r = (*ray).r;
  let theta = (*ray).theta;
  let dr = (*ray).dr;
  let dtheta = (*ray).dtheta;
  let dphi = (*ray).dphi;
  let f = 1.0 - scene.disk.z / max(r, scene.disk.z * 1.01);
  let dtdL = (*ray).energy / max(f, 1e-4);
  let sinTheta = max(sin(theta), 1e-4);
  let cosTheta = cos(theta);
  let d2r = -(scene.disk.z / (2.0 * r * r)) * f * dtdL * dtdL
    + (scene.disk.z / (2.0 * r * r * max(f, 1e-4))) * dr * dr
    + r * (dtheta * dtheta + sinTheta * sinTheta * dphi * dphi);
  let d2theta = -2.0 * dr * dtheta / r + sinTheta * cosTheta * dphi * dphi;
  let d2phi = -2.0 * dr * dphi / r - 2.0 * cosTheta / sinTheta * dtheta * dphi;

  (*ray).r = (*ray).r + dL * dr;
  (*ray).theta = (*ray).theta + dL * dtheta;
  (*ray).phi = (*ray).phi + dL * dphi;
  (*ray).dr = (*ray).dr + dL * d2r;
  (*ray).dtheta = (*ray).dtheta + dL * d2theta;
  (*ray).dphi = (*ray).dphi + dL * d2phi;
  (*ray).x = (*ray).r * sin((*ray).theta) * cos((*ray).phi);
  (*ray).y = (*ray).r * sin((*ray).theta) * sin((*ray).phi);
  (*ray).z = (*ray).r * cos((*ray).theta);
}

fn adaptiveStepScale(r: f32, rs: f32, baseScale: f32) -> f32 {
  if (r > rs * 36.0) {
    return baseScale * 2.4;
  }
  if (r > rs * 18.0) {
    return baseScale * 1.7;
  }
  if (r > rs * 8.0) {
    return baseScale * 1.15;
  }
  return baseScale * 0.72;
}

fn interceptObject(position: vec3f, count: u32) -> ObjectHit {
  for (var i = 0u; i < count; i = i + 1u) {
    let center = objects[i].posRadius.xyz;
    if (distance(position, center) <= objects[i].posRadius.w) {
      return ObjectHit(true, objects[i].color, center);
    }
  }
  return ObjectHit(false, vec4f(0.0), vec3f(0.0));
}

fn crossesEquatorialPlane(oldPos: vec3f, newPos: vec3f) -> bool {
  let crossed = oldPos.y * newPos.y < 0.0;
  let r = length(vec2f(newPos.x, newPos.z));
  return crossed && r >= scene.disk.x && r <= scene.disk.y;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(outImage);
  if (id.x >= size.x || id.y >= size.y) {
    return;
  }

  let uv = (vec2f(id.xy) + vec2f(0.5, 0.5)) / vec2f(size);
  let u = (2.0 * uv.x - 1.0) * camera.params.y * camera.params.x;
  let v = (1.0 - 2.0 * uv.y) * camera.params.x;
  let dir = safeNormalize(u * camera.right.xyz - v * camera.up.xyz + camera.forward.xyz);
  var ray = initRay(camera.pos.xyz, dir);
  var prevPos = vec3f(ray.x, ray.y, ray.z);
  var rmin = 1e30;
  var hitBh = false;
  var hitDisk = false;
  var hitObj = false;
  var objectHit = ObjectHit(false, vec4f(0.0), vec3f(0.0));
  let objectCount = u32(scene.disk.w);
  let stepBudget = max(1u, u32(scene.tuning.x));
  let starLayers = max(1u, u32(scene.tuning.y));
  let stepScale = scene.tuning.z;
  let accumulationBlend = scene.tuning.w;
  let modeLerp = clamp(scene.flags.x / 2.0, 0.0, 1.0);

  for (var step = 0u; step < stepBudget; step = step + 1u) {
    if (ray.r <= scene.disk.z) {
      hitBh = true;
      break;
    }

    let stepSize = D_LAMBDA * adaptiveStepScale(ray.r, scene.disk.z, stepScale);
    geodesicStep(&ray, stepSize);
    rmin = min(rmin, ray.r);
    let newPos = vec3f(ray.x, ray.y, ray.z);
    objectHit = interceptObject(newPos, objectCount);
    if (objectHit.hit) {
      hitObj = true;
      break;
    }
    if (crossesEquatorialPlane(prevPos, newPos)) {
      hitDisk = true;
      break;
    }
    prevPos = newPos;
    if (ray.r > ESCAPE_R) {
      break;
    }
  }

  var color = vec3f(0.0, 0.0, 0.0);
  if (hitDisk) {
    let p = vec3f(ray.x, ray.y, ray.z);
    let rxy = length(vec2f(p.x, p.z));
    let intensity = pow(clamp(scene.disk.y / max(rxy, 1.0), 0.0, 10.0), 2.3);
    let tcol = clamp((rxy - scene.disk.x) / max(scene.disk.y - scene.disk.x, 1.0), 0.0, 1.0);
    let base = heatRamp(1.0 - tcol);
    let phi = atan2(p.z, p.x);
    let omega = 0.35;
    let swirl = 0.5 + 0.5 * sin(6.0 * (phi + omega * camera.params.z));
    let massMax = max(objects[0].velocityMass.w, objects[1].velocityMass.w);
    let beta = keplerBeta(massMax, max(rxy, 1.0));
    let tang = safeNormalize(vec3f(-sin(phi + omega * camera.params.z), 0.0, cos(phi + omega * camera.params.z)));
    let mu = dot(tang, -dir);
    let beaming = pow(dopplerFactor(beta, mu), 3.0);
    let gshift = gravRedshift(rxy, scene.disk.z);
    let edgeIn = smoothstep(scene.disk.x, scene.disk.x + 0.02 * scene.disk.y, rxy);
    let edgeOut = 1.0 - smoothstep(scene.disk.y - 0.02 * scene.disk.y, scene.disk.y, rxy);
    color = base * intensity * mix(0.85, 1.15, swirl) * beaming * gshift * edgeIn * edgeOut;
  } else if (hitBh) {
    color = vec3f(0.0, 0.0, 0.0);
  } else if (hitObj) {
    let p = vec3f(ray.x, ray.y, ray.z);
    let normal = safeNormalize(p - objectHit.center);
    let view = safeNormalize(camera.pos.xyz - p);
    let diffuse = max(dot(normal, view), 0.0);
    color = objectHit.color.rgb * (0.1 + 0.9 * diffuse);
  } else {
    let dirInf = safeNormalize(vec3f(ray.x, ray.y, ray.z));
    let stars = starfield(dirInf, starLayers);
    let sigma = 0.30;
    let x = (rmin / scene.disk.z) - 1.5;
    let boost = exp(-(x * x) / (sigma * sigma)) * mix(2.2, 3.0, modeLerp);
    color = stars * (1.0 + boost);
  }

  color = color / (color + vec3f(1.0, 1.0, 1.0));
  if (accumulationBlend > 0.0) {
    let history = textureLoad(historyImage, vec2<i32>(id.xy), 0).rgb;
    color = mix(color, history, accumulationBlend);
  }

  textureStore(outImage, vec2u(id.xy), vec4f(color, 1.0));
}
