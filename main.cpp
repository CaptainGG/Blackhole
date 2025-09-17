// =================== WINDOWS MACRO FIX ===================
#ifndef NOMINMAX
#define NOMINMAX  // avoid Windows.h min/max macros breaking std::min/std::max
#endif

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <atomic>

// =================== AUDIO (miniaudio) ===================
// Download miniaudio.h from https://miniaud.io and place beside this file.
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace glm;
using namespace std;

// =================== Globals / constants ===================
double c = 299792458.0;
double G = 6.67430e-11;
bool   Gravity = false;

std::atomic<float> gAudioTime {0.0f};
std::atomic<float> gGravBoost{0.0f}; // 0..1 (closer to BH -> louder)

// ------------------- Audio callback -------------------
static void audio_callback(ma_device* dev, void* pOut, const void* /*pIn*/, ma_uint32 frames)
{
    float* out = (float*)pOut;
    static uint32_t rng = 22222;
    static float nLP = 0.0f; // low-pass on noise
    const float sr = (float)dev->sampleRate;

    for (ma_uint32 i = 0; i < frames; ++i) {
        float t = gAudioTime.load(std::memory_order_relaxed);
        gAudioTime.store(t + 1.0f / sr, std::memory_order_relaxed);

        // white noise -> gentle low-pass for airy texture
        rng = 1664525u * rng + 1013904223u;
        float wn = ((rng >> 1) * (1.0f / 2147483647.0f)) * 2.0f - 1.0f;
        nLP += 0.02f * (wn - nLP);

        float grav = gGravBoost.load(std::memory_order_relaxed); // 0..1
        float base = 0.12f + 0.25f * grav;

        float s =
            0.35f * sinf(2.0f * float(M_PI) * 30.0f * t) +
            0.25f * sinf(2.0f * float(M_PI) * 55.0f * t) +
            0.20f * sinf(2.0f * float(M_PI) * (80.0f + 10.0f * grav) * t) +
            0.25f * nLP;

        s *= base;
        float y = tanhf(s);
        out[2*i+0] = y;
        out[2*i+1] = y;
    }
}

// =================== Camera ===================
struct Camera {
    vec3  target     = vec3(0.0f);
    float radius     = 6.34194e10f;
    float minRadius  = 1e10f, maxRadius = 1e12f;

    float azimuth    = 0.0f;
    float elevation  = float(M_PI) / 2.0f;

    float orbitSpeed = 0.01f;
    double zoomSpeed = 25e9f;

    bool   dragging  = false;
    bool   panning   = false;
    bool   moving    = false;
    double lastX     = 0.0, lastY = 0.0;

    vec3 position() const {
        float el = glm::clamp(elevation, 0.01f, float(M_PI) - 0.01f);
        return vec3(
            radius * sin(el) * cos(azimuth),
            radius * cos(el),
            radius * sin(el) * sin(azimuth)
        );
    }

    void update() {
        target = vec3(0.0f);
        moving = (dragging || panning);     // FIX: logical OR
    }

    void processMouseMove(double x, double y) {
        float dx = float(x - lastX);
        float dy = float(y - lastY);

        if (dragging && !panning) {
            azimuth   += dx * orbitSpeed;
            elevation -= dy * orbitSpeed;
            elevation = glm::clamp(elevation, 0.01f, float(M_PI) - 0.01f);
        }
        lastX = x; lastY = y;
        update();
    }

    void processMouseButton(int button, int action, int mods, GLFWwindow* win) {
        if (button == GLFW_MOUSE_BUTTON_LEFT || button == GLFW_MOUSE_BUTTON_MIDDLE) {
            if (action == GLFW_PRESS) {
                dragging = true; panning = false;
                glfwGetCursorPos(win, &lastX, &lastY);
            } else if (action == GLFW_RELEASE) {
                dragging = false; panning = false;
            }
        }
        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            Gravity = (action == GLFW_PRESS);
        }
    }

    void processScroll(double, double yoffset) {
        radius -= yoffset * zoomSpeed;
        radius = glm::clamp(radius, minRadius, maxRadius);
        update();
    }

    void processKey(int key, int, int action, int) {
        if (action == GLFW_PRESS && key == GLFW_KEY_G) {
            Gravity = !Gravity;
            std::cout << "[INFO] Gravity " << (Gravity ? "ON" : "OFF") << "\n";
        }
    }
};
Camera camera;

// =================== Scene objects ===================
struct BlackHole {
    vec3   position;
    double mass;
    double r_s;
    BlackHole(vec3 pos, double m) : position(pos), mass(m) {
        r_s = 2.0 * G * mass / (c * c);
    }
};
BlackHole SagA(vec3(0.0f), 8.54e36); // Sagittarius A*

struct ObjectData {
    vec4  posRadius; // xyz position, w radius
    vec4  color;     // rgb color
    double mass;     // CPU precision for physics
    vec3  velocity = vec3(0.0f);
    float _pad = 0.0f;
};

vector<ObjectData> objects = {
    { vec4(4e11f, 0.0f, 0.0f, 4e10f), vec4(1,1,0,1), 1.98892e30 },
    { vec4(0.0f, 0.0f, 4e11f, 4e10f), vec4(1,0,0,1), 1.98892e30 },
  // NEW spheres (note non-zero Y and varied sizes)
    { vec4(-4e11f,  1.5e11f, -2e11f, 4.5e10f), vec4(0.20f, 0.80f, 1.00f, 1.0f), 0.0 }, // cyan, above disk
    { vec4( 2e11f,  2.5e11f, -6e11f, 2.8e10f), vec4(0.85f, 0.20f, 1.00f, 1.0f), 0.0 }, // magenta, higher
    { vec4(-6e11f, -1.2e11f,  3e11f, 5.5e10f), vec4(0.20f, 1.00f, 0.35f, 1.0f), 0.0 }, // green, below disk
	
    { vec4(0.0f, 0.0f, 0.0f, float(SagA.r_s)), vec4(0,0,0,1), SagA.mass },
};

// =================== Engine (OpenGL) ===================
struct Engine {
    GLFWwindow* window = nullptr;

    GLuint shaderProgram = 0;      // fullscreen quad shader
    GLuint computeProgram = 0;     // raytracer
    GLuint gridShaderProgram = 0;

    GLuint quadVAO = 0;
    GLuint texture = 0;

    // UBOs
    GLuint cameraUBO  = 0; // binding = 1
    GLuint diskUBO    = 0; // binding = 2
    GLuint objectsUBO = 0; // binding = 3

    // Grid buffers
    GLuint gridVAO = 0, gridVBO = 0, gridEBO = 0;
    int    gridIndexCount = 0;

    // Window & compute sizes
    int WIDTH  = 800;
    int HEIGHT = 600;
    int COMPUTE_WIDTH  = 200;
    int COMPUTE_HEIGHT = 150;
    int lastComputeW = 0, lastComputeH = 0;

    Engine() {
        if (!glfwInit()) { std::cerr << "GLFW init failed\n"; exit(EXIT_FAILURE); }
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Black Hole", nullptr, nullptr);
        if (!window) { std::cerr << "Window creation failed\n"; glfwTerminate(); exit(EXIT_FAILURE); }
        glfwMakeContextCurrent(window);

        glewExperimental = GL_TRUE;
        if (glewInit() != GLEW_OK) { std::cerr << "GLEW init failed\n"; exit(EXIT_FAILURE); }
        std::cout << "OpenGL " << glGetString(GL_VERSION) << "\n";

        shaderProgram     = CreateShaderProgram();
        gridShaderProgram = CreateShaderProgram("grid.vert", "grid.frag");
        computeProgram    = CreateComputeProgram("geodesic.comp");

        // UBOs
        glGenBuffers(1, &cameraUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, cameraUBO);
        glBufferData(GL_UNIFORM_BUFFER, 128, nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 1, cameraUBO);

        glGenBuffers(1, &diskUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, diskUBO);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(float)*4, nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 2, diskUBO);

        glGenBuffers(1, &objectsUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, objectsUBO);
        GLsizeiptr objUBOSize =
            sizeof(int) + 3*sizeof(float) +                 // header padding
            16*(sizeof(vec4)+sizeof(vec4)) +                // posRadius + color
            16*sizeof(vec4);                                 // masses as vec4
        glBufferData(GL_UNIFORM_BUFFER, objUBOSize, nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 3, objectsUBO);

        auto q = QuadVAO();
        quadVAO = q[0];
        texture = q[1];

        // on resize, update viewport
        glfwSetFramebufferSizeCallback(window, [](GLFWwindow*, int w, int h){
            if (w > 0 && h > 0) glViewport(0, 0, w, h);
        });
    }

    // ---- Grid (deform for visualization only) ----
void generateGrid(const vector<ObjectData>& objs) {
    const int   gridSize = 25;
    const float spacing  = 1e10f;

    vector<vec3> verts;
    vector<GLuint> idx;
    verts.reserve((gridSize+1)*(gridSize+1));
    idx.reserve(gridSize*gridSize*4);

    for (int z=0; z<=gridSize; ++z) {
        for (int x=0; x<=gridSize; ++x) {
            float wx = (x - gridSize/2) * spacing;
            float wz = (z - gridSize/2) * spacing;

            // baseline once (not per object):
            float y = -3e10f;

            for (const auto& o : objs) {
                if (o.mass <= 0.0) continue;            // <-- skip massless visuals

                vec3   pos = vec3(o.posRadius);
                double rs  = 2.0 * G * o.mass / (c*c);
                double dx  = wx - pos.x, dz = wz - pos.z;
                double dist = sqrt(dx*dx + dz*dz);

                if (dist > rs)      y += float(2.0 * sqrt(rs * (dist - rs)));
                else /* dist<=rs */ y += float(2.0 * sqrt(rs * rs)); // deep pit
            }
            verts.emplace_back(wx, y, wz);
        }
    }

    for (int z=0; z<gridSize; ++z) for (int x=0; x<gridSize; ++x) {
        int i = z*(gridSize+1)+x;
        idx.push_back(i);   idx.push_back(i+1);
        idx.push_back(i);   idx.push_back(i+gridSize+1);
    }

    if (!gridVAO) glGenVertexArrays(1,&gridVAO);
    if (!gridVBO) glGenBuffers(1,&gridVBO);
    if (!gridEBO) glGenBuffers(1,&gridEBO);

    glBindVertexArray(gridVAO);
    glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
    glBufferData(GL_ARRAY_BUFFER, verts.size()*sizeof(vec3), verts.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gridEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size()*sizeof(GLuint), idx.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(vec3),(void*)0);
    gridIndexCount = (int)idx.size();
    glBindVertexArray(0);
}



    void drawGrid(const mat4& vp) {
        glUseProgram(gridShaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(gridShaderProgram,"viewProj"),1,GL_FALSE,value_ptr(vp));
        glBindVertexArray(gridVAO);
        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDrawElements(GL_LINES, gridIndexCount, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
		glLineWidth(1.5f); // some drivers clamp to 1.0, but it helps where supported
        glEnable(GL_DEPTH_TEST);
    }

    // ---- Fullscreen quad ----
    vector<GLuint> QuadVAO(){
        float quadVertices[] = {
            // positions   // texCoords
            -1.0f,  1.0f,  0.0f, 1.0f,
            -1.0f, -1.0f,  0.0f, 0.0f,
             1.0f, -1.0f,  1.0f, 0.0f,

            -1.0f,  1.0f,  0.0f, 1.0f,
             1.0f, -1.0f,  1.0f, 0.0f,
             1.0f,  1.0f,  1.0f, 1.0f
        };
        GLuint VAO, VBO;
        glGenVertexArrays(1,&VAO);
        glGenBuffers(1,&VBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
        glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)(2*sizeof(float)));
        glEnableVertexAttribArray(1);

        GLuint tex;
        glGenTextures(1,&tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, COMPUTE_WIDTH, COMPUTE_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        return {VAO, tex};
    }

    void drawFullScreenQuad() {
        glUseProgram(shaderProgram);
        glBindVertexArray(quadVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glUniform1i(glGetUniformLocation(shaderProgram, "screenTexture"), 0);
        glDisable(GL_DEPTH_TEST);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 6);
        glEnable(GL_DEPTH_TEST);
    }

    // ---- Shaders ----
    GLuint CreateShaderProgram(){
        const char* vs = R"(
            #version 330 core
            layout(location=0) in vec2 aPos;
            layout(location=1) in vec2 aTex;
            out vec2 uv;
            void main(){ gl_Position=vec4(aPos,0,1); uv=aTex; }
        )";
        const char* fs = R"(
            #version 330 core
            in vec2 uv;
            out vec4 FragColor;
            uniform sampler2D screenTexture;
            void main(){ FragColor = texture(screenTexture, uv); }
        )";
        GLuint v = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(v,1,&vs,nullptr); glCompileShader(v);
        GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(f,1,&fs,nullptr); glCompileShader(f);
        GLuint p = glCreateProgram();
        glAttachShader(p,v); glAttachShader(p,f); glLinkProgram(p);
        glDeleteShader(v); glDeleteShader(f);
        return p;
    }

    GLuint CreateShaderProgram(const char* vertPath, const char* fragPath) {
        auto load = [](const char* path, GLenum type){
            std::ifstream in(path);
            if(!in.is_open()){ std::cerr<<"Failed to open "<<path<<"\n"; exit(EXIT_FAILURE); }
            std::stringstream ss; ss<<in.rdbuf(); std::string s=ss.str(); const char* src=s.c_str();
            GLuint sh = glCreateShader(type);
            glShaderSource(sh,1,&src,nullptr); glCompileShader(sh);
            GLint ok; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
            if(!ok){ GLint n; glGetShaderiv(sh, GL_INFO_LOG_LENGTH,&n); std::vector<char> log(n);
                glGetShaderInfoLog(sh,n,nullptr,log.data()); std::cerr<<"Compile "<<path<<":\n"<<log.data(); exit(EXIT_FAILURE); }
            return sh;
        };
        GLuint vs = load(vertPath, GL_VERTEX_SHADER);
        GLuint fs = load(fragPath, GL_FRAGMENT_SHADER);
        GLuint p = glCreateProgram();
        glAttachShader(p,vs); glAttachShader(p,fs); glLinkProgram(p);
        GLint ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if(!ok){ GLint n; glGetProgramiv(p, GL_INFO_LOG_LENGTH,&n); std::vector<char> log(n);
            glGetProgramInfoLog(p,n,nullptr,log.data()); std::cerr<<"Link error:\n"<<log.data(); exit(EXIT_FAILURE); }
        glDeleteShader(vs); glDeleteShader(fs);
        return p;
    }

    GLuint CreateComputeProgram(const char* path) {
        std::ifstream in(path);
        if(!in.is_open()){ std::cerr<<"Failed to open "<<path<<"\n"; exit(EXIT_FAILURE); }
        std::stringstream ss; ss<<in.rdbuf(); std::string s=ss.str(); const char* src=s.c_str();
        GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
        glShaderSource(cs,1,&src,nullptr); glCompileShader(cs);
        GLint ok; glGetShaderiv(cs, GL_COMPILE_STATUS, &ok);
        if(!ok){ GLint n; glGetShaderiv(cs, GL_INFO_LOG_LENGTH,&n); std::vector<char> log(n);
            glGetShaderInfoLog(cs,n,nullptr,log.data()); std::cerr<<"Compute compile:\n"<<log.data(); exit(EXIT_FAILURE); }
        GLuint p = glCreateProgram();
        glAttachShader(p,cs); glLinkProgram(p);
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if(!ok){ GLint n; glGetProgramiv(p, GL_INFO_LOG_LENGTH,&n); std::vector<char> log(n);
            glGetProgramInfoLog(p,n,nullptr,log.data()); std::cerr<<"Compute link:\n"<<log.data(); exit(EXIT_FAILURE); }
        glDeleteShader(cs);
        return p;
    }

    // ---- Dispatch compute pass ----
    void dispatchCompute(const Camera& cam) {
        int cw = cam.moving ? COMPUTE_WIDTH  : 200;
        int ch = cam.moving ? COMPUTE_HEIGHT : 150;

        glBindTexture(GL_TEXTURE_2D, texture);
        if (cw != lastComputeW || ch != lastComputeH) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, cw, ch, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            lastComputeW = cw; lastComputeH = ch;
        }

        glUseProgram(computeProgram);

        // Pass time uniform for animation
        GLint uTimeLoc = glGetUniformLocation(computeProgram, "uTime");
        if (uTimeLoc >= 0) glUniform1f(uTimeLoc, float(glfwGetTime()));

        uploadCameraUBO(cam, cw, ch);
        uploadDiskUBO();
        uploadObjectsUBO(objects);

        glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);

        GLuint gx = (GLuint)std::ceil(cw / 16.0f);
        GLuint gy = (GLuint)std::ceil(ch / 16.0f);
        glDispatchCompute(gx, gy, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    // ---- UBO uploads ----
    void uploadCameraUBO(const Camera& cam, int texW, int texH) {
        struct UBO {
            vec3 pos;     float _p0;
            vec3 right;   float _p1;
            vec3 up;      float _p2;
            vec3 forward; float _p3;
            float tanHalfFov;
            float aspect;
            int   moving;
            int   _p4;
        } data;

        vec3 fwd = normalize(cam.target - cam.position());
        vec3 up  = vec3(0,1,0);
        vec3 right = normalize(cross(fwd, up));
        up = cross(right, fwd);

        data.pos = cam.position();
        data.right = right;
        data.up = up;
        data.forward = fwd;
        data.tanHalfFov = tan(radians(60.0f * 0.5f));
        data.aspect = float(texW) / float(texH);     // aspect of compute texture
        data.moving = (cam.dragging || cam.panning) ? 1 : 0;

        glBindBuffer(GL_UNIFORM_BUFFER, cameraUBO);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(UBO), &data);
    }

    void uploadObjectsUBO(const vector<ObjectData>& objs) {
        struct UBO {
            int   numObjects; float _p0, _p1, _p2;
            vec4  posRadius[16];
            vec4  color[16];
            vec4  mass[16]; // use .x in shader
        } data{};

        size_t n = (std::min)(objs.size(), size_t(16)); // macro-safe
        data.numObjects = int(n);
        for (size_t i=0;i<n;++i) {
            data.posRadius[i] = objs[i].posRadius;
            data.color[i]     = objs[i].color;
            data.mass[i]      = vec4(float(objs[i].mass), 0,0,0);
        }

        glBindBuffer(GL_UNIFORM_BUFFER, objectsUBO);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(UBO), &data);
    }

    void uploadDiskUBO() {
        float r1 = float(SagA.r_s * 2.2);
        float r2 = float(SagA.r_s * 5.2);
        float num = 2.0f, thick = 1e9f;
        float disk[4] = { r1, r2, num, thick };
        glBindBuffer(GL_UNIFORM_BUFFER, diskUBO);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(disk), disk);
    }
};
Engine engine;

// =================== Input setup ===================
void setupCameraCallbacks(GLFWwindow* window) {
    glfwSetWindowUserPointer(window, &camera);

    glfwSetMouseButtonCallback(window, [](GLFWwindow* win, int button, int action, int mods) {
        ((Camera*)glfwGetWindowUserPointer(win))->processMouseButton(button, action, mods, win);
    });
    glfwSetCursorPosCallback(window, [](GLFWwindow* win, double x, double y) {
        ((Camera*)glfwGetWindowUserPointer(win))->processMouseMove(x, y);
    });
    glfwSetScrollCallback(window, [](GLFWwindow* win, double xoff, double yoff) {
        ((Camera*)glfwGetWindowUserPointer(win))->processScroll(xoff, yoff);
    });
    glfwSetKeyCallback(window, [](GLFWwindow* win, int key, int sc, int action, int mods) {
        ((Camera*)glfwGetWindowUserPointer(win))->processKey(key, sc, action, mods);
    });
}

// =================== MAIN ===================
int main() {
    setupCameraCallbacks(engine.window);

    // ---- start audio ----
    ma_device_config cfg = ma_device_config_init(ma_device_type_playback);
    cfg.playback.format   = ma_format_f32;
    cfg.playback.channels = 2;
    cfg.sampleRate        = 48000;
    cfg.dataCallback      = audio_callback;

    ma_device device; // keep in scope of main
    bool audioOK = (ma_device_init(nullptr, &cfg, &device) == MA_SUCCESS);
    if (audioOK) {
        ma_device_start(&device);
    } else {
        std::cerr << "[WARN] Audio init failed; continuing muted.\n";
    }

    double lastTime = glfwGetTime();
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    while (!glfwWindowShouldClose(engine.window)) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        double now = glfwGetTime();
        double dt  = now - lastTime;
        lastTime   = now;

        // ---- Idle auto-orbit animation ----
        if (!camera.dragging && !camera.panning) {
            camera.azimuth += 0.15f * float(dt);   // radians/sec
            camera.update();
        }

        // ---- Drive audio intensity by camera proximity to BH (closer -> louder) ----
        float camR = length(camera.position());
        float nearR = float(SagA.r_s * 6.0);
        float farR  = float(SagA.r_s * 60.0);
        float norm  = (camR - nearR) / (farR - nearR);
        float gravBoost = 1.0f - glm::clamp(norm, 0.0f, 1.0f);  // disambiguate clamp
        gGravBoost.store(gravBoost, std::memory_order_relaxed);
        gAudioTime.store(float(now), std::memory_order_relaxed);

        // ---- Simple N-body (when enabled) ----
        if (Gravity) {
            const double softening2 = 1e20;
            for (size_t i=0;i<objects.size();++i) {
                for (size_t j=i+1;j<objects.size();++j) {
                    vec3 rij = vec3(objects[j].posRadius) - vec3(objects[i].posRadius);
                    double r2 = double(dot(rij,rij)) + softening2;
                    double invR = 1.0 / sqrt(r2);
                    double invR3 = invR * invR * invR;
                    double f = G * objects[i].mass * objects[j].mass * invR3;
                    vec3 ai =  float( f / objects[i].mass) * rij;
                    vec3 aj = -float( f / objects[j].mass) * rij;
                    objects[i].velocity += ai * float(dt);
                    objects[j].velocity += aj * float(dt);
                }
            }
            for (auto& o : objects) o.posRadius += vec4(o.velocity * float(dt), 0.0f);
        }

        // ---- Grid (visual) ----
        engine.generateGrid(objects);
		engine.dispatchCompute(camera);
		engine.drawFullScreenQuad();
		
        mat4 view = lookAt(camera.position(), camera.target, vec3(0,1,0));
        mat4 proj = perspective(radians(60.0f), float(engine.WIDTH)/float(engine.HEIGHT), 1e9f, 1e14f);
        mat4 vp = proj * view;
		engine.drawGrid(vp);

        // ---- Compute raytracer ----
        glViewport(0,0,engine.WIDTH,engine.HEIGHT);
        engine.dispatchCompute(camera);
        engine.drawFullScreenQuad();

        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }

    // ---- stop audio ----
    if (audioOK) {
        ma_device_uninit(&device);
    }

    glfwDestroyWindow(engine.window);
    glfwTerminate();
    return 0;
}
