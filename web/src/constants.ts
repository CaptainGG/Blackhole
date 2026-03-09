import type { ObjectData, Vec3 } from "./types";

export const C = 299792458.0;
export const G = 6.67430e-11;
export const BLACK_HOLE_MASS = 8.54e36;
export const BLACK_HOLE_RS = (2.0 * G * BLACK_HOLE_MASS) / (C * C);
export const GRID_SIZE = 25;
export const GRID_SPACING = 1e10;

function vec3(x: number, y: number, z: number): Vec3 {
  return { x, y, z };
}

export const INITIAL_OBJECTS: ObjectData[] = [
  {
    position: vec3(4e11, 0.0, 0.0),
    radius: 4e10,
    color: [1, 1, 0, 1],
    mass: 1.98892e30,
    velocity: vec3(0, 0, 0)
  },
  {
    position: vec3(0.0, 0.0, 4e11),
    radius: 4e10,
    color: [1, 0, 0, 1],
    mass: 1.98892e30,
    velocity: vec3(0, 0, 0)
  },
  {
    position: vec3(-4e11, 1.5e11, -2e11),
    radius: 4.5e10,
    color: [0.2, 0.8, 1.0, 1.0],
    mass: 0.0,
    velocity: vec3(0, 0, 0)
  },
  {
    position: vec3(2e11, 2.5e11, -6e11),
    radius: 2.8e10,
    color: [0.85, 0.2, 1.0, 1.0],
    mass: 0.0,
    velocity: vec3(0, 0, 0)
  },
  {
    position: vec3(-6e11, -1.2e11, 3e11),
    radius: 5.5e10,
    color: [0.2, 1.0, 0.35, 1.0],
    mass: 0.0,
    velocity: vec3(0, 0, 0)
  },
  {
    position: vec3(0.0, 0.0, 0.0),
    radius: BLACK_HOLE_RS,
    color: [0, 0, 0, 1],
    mass: BLACK_HOLE_MASS,
    velocity: vec3(0, 0, 0)
  }
];
