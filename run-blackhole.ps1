param(
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$buildDir = Join-Path $root "build-vs"
$toolchain = Join-Path $root "vcpkg\scripts\buildsystems\vcpkg.cmake"
$exe = Join-Path $buildDir "$Configuration\black_hole.exe"

if (-not (Test-Path $toolchain)) {
    throw "vcpkg was not found at $toolchain. Build the project setup first."
}

if (-not (Test-Path (Join-Path $buildDir "black_hole.sln"))) {
    cmake -S $root -B $buildDir -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=$toolchain -DVCPKG_TARGET_TRIPLET=x64-windows
}

cmake --build $buildDir --config $Configuration

if (-not (Test-Path $exe)) {
    throw "Expected executable not found at $exe"
}

Start-Process -FilePath $exe -WorkingDirectory (Split-Path -Parent $exe)
