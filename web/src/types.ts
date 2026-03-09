export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface ObjectData {
  position: Vec3;
  radius: number;
  color: [number, number, number, number];
  mass: number;
  velocity: Vec3;
}

export interface BlackHoleData {
  position: Vec3;
  mass: number;
  schwarzschildRadius: number;
}

export interface CameraFrame {
  position: Vec3;
  forward: Vec3;
  right: Vec3;
  up: Vec3;
}
