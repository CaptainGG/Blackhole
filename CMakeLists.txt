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

target_compile_definitions(black_hole PRIVATE BLACKHOLE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(black_hole PRIVATE
  OpenGL::GL
  glfw
  GLEW::GLEW
  glm::glm
)

# Copy shaders next to the final executable so it can be launched from Release/.
foreach(f grid.vert grid.frag geodesic.comp)
  add_custom_command(TARGET black_hole POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_SOURCE_DIR}/${f}
      $<TARGET_FILE_DIR:black_hole>/${f}
  )
endforeach()

# Bundle runtime DLLs from vcpkg/OpenGL dependencies beside the executable.
add_custom_command(TARGET black_hole POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_RUNTIME_DLLS:black_hole>
    $<TARGET_FILE_DIR:black_hole>
  COMMAND_EXPAND_LISTS
)

