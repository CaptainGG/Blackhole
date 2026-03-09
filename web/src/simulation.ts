import {
  BLACK_HOLE_MASS,
  BLACK_HOLE_RS,
  C,
  G,
  GRID_SIZE,
  GRID_SPACING,
  INITIAL_OBJECTS
} from "./constants";
import {
  add,
  clamp,
  cloneVec3,
  cross,
  length,
  normalize,
  perspectiveMatrix,
  lookAtMatrix,
  multiplyMatrices,
  scale,
  subtract,
  vec3
} from "./math";
import type { BlackHoleData, CameraFrame, ObjectData, Vec3 } from "./types";

export type RenderQualityMode = "motion" | "settling" | "idle";

const INTERACTION_SETTLE_SECONDS = 0.18;
const ZOOM_EPSILON = 2.5e6;

export class Camera {
  target: Vec3 = vec3(0, 0, 0);
  radius = 6.34194e10;
  minRadius = 1e10;
  maxRadius = 1e12;
  azimuth = 0;
  elevation = Math.PI / 2;
  dragging = false;
  moving = false;
  isInteracting = false;
  settling = false;
  lastInteractionTime = 0;
  lastX = 0;
  lastY = 0;
  zoomTarget = this.radius;
  zoomVelocity = 0;

  position(): Vec3 {
    const el = clamp(this.elevation, 0.01, Math.PI - 0.01);
    return vec3(
      this.radius * Math.sin(el) * Math.cos(this.azimuth),
      this.radius * Math.cos(el),
      this.radius * Math.sin(el) * Math.sin(this.azimuth)
    );
  }

  frame(): CameraFrame {
    const position = this.position();
    const forward = normalize(subtract(this.target, position));
    const right = normalize(cross(forward, vec3(0, 1, 0)));
    const up = cross(right, forward);
    return { position, forward, right, up };
  }

  beginDrag(x: number, y: number, time: number): void {
    this.dragging = true;
    this.lastX = x;
    this.lastY = y;
    this.markInteraction(time);
  }

  endDrag(time: number): void {
    this.dragging = false;
    this.markInteraction(time);
  }

  pointerMove(x: number, y: number, viewportWidth: number, viewportHeight: number, time: number): void {
    if (!this.dragging) {
      return;
    }

    const dx = x - this.lastX;
    const dy = y - this.lastY;
    const width = Math.max(320, viewportWidth);
    const height = Math.max(240, viewportHeight);
    const orbitScaleX = (Math.PI * 1.55) / width;
    const orbitScaleY = (Math.PI * 1.25) / height;

    this.azimuth += dx * orbitScaleX;
    this.elevation = clamp(this.elevation - dy * orbitScaleY, 0.01, Math.PI - 0.01);
    this.lastX = x;
    this.lastY = y;
    this.markInteraction(time);
  }

  zoom(deltaPixels: number, time: number): void {
    const zoomFactor = Math.exp(deltaPixels * 0.0012);
    this.zoomTarget = clamp(this.zoomTarget * zoomFactor, this.minRadius, this.maxRadius);
    this.markInteraction(time);
  }

  update(dt: number, time: number): void {
    const diff = this.zoomTarget - this.radius;
    const blend = 1.0 - Math.exp(-20.0 * dt);
    const radiusDelta = diff * blend;
    this.zoomVelocity = radiusDelta / Math.max(dt, 1e-4);
    this.radius = clamp(this.radius + radiusDelta, this.minRadius, this.maxRadius);

    if (Math.abs(this.zoomTarget - this.radius) < ZOOM_EPSILON) {
      this.radius = this.zoomTarget;
      this.zoomVelocity = 0;
    }

    const hasZoomMotion = Math.abs(this.zoomTarget - this.radius) >= ZOOM_EPSILON;
    this.settling = !this.dragging && (hasZoomMotion || time - this.lastInteractionTime < INTERACTION_SETTLE_SECONDS);
    this.isInteracting = this.dragging || this.settling;
    this.moving = this.isInteracting;
  }

  renderQuality(): RenderQualityMode {
    if (this.dragging) {
      return "motion";
    }
    if (this.settling) {
      return "settling";
    }
    return "idle";
  }

  private markInteraction(time: number): void {
    this.lastInteractionTime = time;
    this.isInteracting = true;
    this.moving = true;
  }
}

export class Simulation {
  readonly blackHole: BlackHoleData = {
    position: vec3(0, 0, 0),
    mass: BLACK_HOLE_MASS,
    schwarzschildRadius: BLACK_HOLE_RS
  };

  readonly camera = new Camera();
  readonly objects: ObjectData[] = INITIAL_OBJECTS.map((object) => ({
    ...object,
    position: cloneVec3(object.position),
    velocity: cloneVec3(object.velocity),
    color: [...object.color] as [number, number, number, number]
  }));

  gravityEnabled = false;
  time = 0;
  gravBoost = 0;
  private gridVerticesCache: Float32Array;
  private readonly gridIndicesCache: Uint32Array;
  private objectDataCache = new Float32Array(16 * 12);
  private gridDirty = true;
  private objectDataDirty = true;

  constructor() {
    this.gridIndicesCache = this.buildGridIndices();
    this.gridVerticesCache = this.buildGridVertices();
  }

  update(dt: number): void {
    this.time += dt;
    this.camera.update(dt, this.time);

    if (!this.camera.isInteracting) {
      this.camera.azimuth += 0.15 * dt;
    }

    const cameraRadius = length(this.camera.position());
    const nearRadius = this.blackHole.schwarzschildRadius * 6.0;
    const farRadius = this.blackHole.schwarzschildRadius * 60.0;
    const norm = (cameraRadius - nearRadius) / (farRadius - nearRadius);
    this.gravBoost = 1.0 - clamp(norm, 0.0, 1.0);

    if (this.gravityEnabled) {
      this.stepGravity(dt);
    }
  }

  toggleGravity(): void {
    this.gravityEnabled = !this.gravityEnabled;
  }

  consumeGridVertices(): Float32Array | null {
    if (!this.gridDirty) {
      return null;
    }
    this.gridVerticesCache = this.buildGridVertices();
    this.gridDirty = false;
    return this.gridVerticesCache;
  }

  getGridVertices(): Float32Array {
    return this.gridVerticesCache;
  }

  getGridIndices(): Uint32Array {
    return this.gridIndicesCache;
  }

  consumeObjectData(): Float32Array | null {
    if (!this.objectDataDirty) {
      return null;
    }

    this.objectDataCache.fill(0);
    this.objects.slice(0, 16).forEach((object, index) => {
      const offset = index * 12;
      this.objectDataCache.set([
        object.position.x,
        object.position.y,
        object.position.z,
        object.radius,
        object.color[0],
        object.color[1],
        object.color[2],
        object.color[3],
        object.velocity.x,
        object.velocity.y,
        object.velocity.z,
        object.mass
      ], offset);
    });

    this.objectDataDirty = false;
    return this.objectDataCache;
  }

  viewProjection(aspect: number): Float32Array {
    const frame = this.camera.frame();
    const view = lookAtMatrix(frame.position, this.camera.target, vec3(0, 1, 0));
    const projection = perspectiveMatrix((60 * Math.PI) / 180, Math.max(aspect, 0.0001), 1e9, 1e14);
    return multiplyMatrices(projection, view);
  }

  private stepGravity(dt: number): void {
    const softening2 = 1e20;
    for (let i = 0; i < this.objects.length; i += 1) {
      for (let j = i + 1; j < this.objects.length; j += 1) {
        const a = this.objects[i];
        const b = this.objects[j];
        if (a.mass <= 0 || b.mass <= 0) {
          continue;
        }

        const delta = subtract(b.position, a.position);
        const r2 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z + softening2;
        const invR = 1.0 / Math.sqrt(r2);
        const invR3 = invR * invR * invR;
        const force = G * a.mass * b.mass * invR3;
        const ai = scale(delta, force / a.mass);
        const aj = scale(delta, -force / b.mass);
        a.velocity = add(a.velocity, scale(ai, dt));
        b.velocity = add(b.velocity, scale(aj, dt));
      }
    }

    for (const object of this.objects) {
      object.position = add(object.position, scale(object.velocity, dt));
    }

    this.gridDirty = true;
    this.objectDataDirty = true;
  }

  private buildGridVertices(): Float32Array {
    const vertices: number[] = [];
    for (let z = 0; z <= GRID_SIZE; z += 1) {
      for (let x = 0; x <= GRID_SIZE; x += 1) {
        const wx = (x - GRID_SIZE / 2) * GRID_SPACING;
        const wz = (z - GRID_SIZE / 2) * GRID_SPACING;
        let y = -3e10;

        for (const object of this.objects) {
          if (object.mass <= 0) {
            continue;
          }

          const rs = (2.0 * G * object.mass) / (C * C);
          const dx = wx - object.position.x;
          const dz = wz - object.position.z;
          const dist = Math.sqrt(dx * dx + dz * dz);
          y += dist > rs ? 2.0 * Math.sqrt(rs * (dist - rs)) : 2.0 * Math.sqrt(rs * rs);
        }

        vertices.push(wx, y, wz);
      }
    }
    return new Float32Array(vertices);
  }

  private buildGridIndices(): Uint32Array {
    const indices: number[] = [];
    for (let z = 0; z < GRID_SIZE; z += 1) {
      for (let x = 0; x < GRID_SIZE; x += 1) {
        const i = z * (GRID_SIZE + 1) + x;
        indices.push(i, i + 1, i, i + GRID_SIZE + 1);
      }
    }
    return new Uint32Array(indices);
  }
}
