## Blackhole V2

Blackhole V2 keeps the original native OpenGL black hole demo and adds a shareable WebGPU browser experience that preserves the same core feel: relativistic lensing, an accretion disk, orbit controls, a warped gravity grid, and a gravity toggle.

It now includes:
- A hardened native desktop app that launches more reliably on Windows
- A WebGPU-based web demo in `web/` for sharing the experience in modern desktop browsers
- A simple deployment path for the web build with `vercel.json`

The project is still centered on graphics programming and simulation, but it now ships in two forms:
- **Native desktop demo** for the original OpenGL compute-shader experience
- **Web demo** for a browser-based version that can be shared publicly

---

## Native Desktop Demo

The native version is a C++ / OpenGL application built with CMake and `vcpkg`.

### Native features
- GPU-accelerated black hole rendering with a GLSL compute shader
- Startup checks for missing shaders or unsupported graphics capabilities
- A helper launcher script for building and running the app on Windows

### Run the native app

From the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File .\run-blackhole.ps1
```

This script reuses the Visual Studio build directory if present, rebuilds `Release` when needed, and launches the native executable with the correct working directory.

---

## Web Demo

The `web/` folder contains a Vite + TypeScript + WebGPU port of the simulation for modern desktop Chrome and Edge.

### Web features
- WebGPU black hole renderer with adaptive quality modes
- Orbit drag, wheel zoom, gravity toggle, and motion-aware HUD behavior
- Performance smoothing for interaction-heavy moments
- Procedural ambient audio unlocked after user interaction

### Run the web demo locally

```powershell
cd .\web
npm.cmd install
npm.cmd run dev
```

### Build the web demo

```powershell
cd .\web
npm.cmd run build
```

---

## Deployment

The repo includes `vercel.json` so the web client can be deployed as a static site directly from this repository.

Expected Vercel flow:
- install command: `npm install --prefix web`
- build command: `npm run build --prefix web`
- output directory: `web/dist`

---

## Requirements

### Native
- C++17-compatible compiler
- CMake
- `vcpkg`
- Git
- OpenGL 4.3 compatible GPU / drivers

### Web
- Node.js with npm
- Desktop Chrome or Edge with WebGPU support

---

## Project Layout

- `main.cpp` and `geodesic.comp` power the native renderer
- `run-blackhole.ps1` launches the native app on Windows
- `web/` contains the browser version
- `vercel.json` configures static deployment for the web app
