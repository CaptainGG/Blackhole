#version 330 core
out vec4 FragColor;

// Optional uniform if you want to tweak color from C++
uniform vec4 uColor = vec4(0.60, 0.85, 1.00, 0.45);

void main() {
    FragColor = uColor;  // transparent cyan
}
