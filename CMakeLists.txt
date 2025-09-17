cmake_minimum_required(VERSION 3.20)
project(black_hole LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages via vcpkg (CONFIG mode)
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

add_executable(black_hole main.cpp)

target_link_libraries(black_hole PRIVATE
  OpenGL::GL
  glfw
  GLEW::GLEW
  glm::glm
)

# Copy shaders next to the binary so the app can open them by filename
foreach(f grid.vert grid.frag geodesic.comp)
  configure_file(${CMAKE_SOURCE_DIR}/${f} ${CMAKE_BINARY_DIR}/${f} COPYONLY)
endforeach()
