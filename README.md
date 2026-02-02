## Overview

This project simulates how light bends around a black hole due to extreme gravitational effects.

It includes:
- A **2D gravitational lensing** implementation for conceptual clarity  
- A **3D GPU-accelerated simulation** using compute shaders for realistic, high-performance results  
- A modern **CMake + vcpkg** build system  

The goal of this project is to demonstrate strong fundamentals in **graphics programming**, **simulation**, and **modern C++ tooling**.

---

## 3D Simulation

The 3D version is optimized for performance using the GPU:

- Core simulation logic is implemented in `black_hole.cpp`
- Heavy numerical computations are executed in a GLSL compute shader (`geodesic.comp`)
- Simulation parameters are sent to the GPU via a **Uniform Buffer Object (UBO)**
- Light geodesics are computed in parallel on the GPU, significantly improving performance

This hybrid CPU/GPU architecture enables more complex and realistic simulations than a purely CPU-based approach.

---

## Build & Run

### Requirements
- C++17-compatible compiler  
- CMake  
- vcpkg  
- Git  
