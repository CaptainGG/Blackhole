import fullscreenWGSL from "./shaders/fullscreen.wgsl?raw";
import gridWGSL from "./shaders/grid.wgsl?raw";
import computeWGSL from "./shaders/blackhole.wgsl?raw";
import { BLACK_HOLE_RS } from "./constants";
import type { RenderQualityMode, Simulation } from "./simulation";

type QualityProfile = {
  scale: number;
  maxDpr: number;
  minWidth: number;
  minHeight: number;
  stepBudget: number;
  starLayerCount: number;
  stepScale: number;
  accumulationBlend: number;
  modeFlag: number;
};

const MAX_OBJECTS = 16;
const EMA_ALPHA = 0.15;
const SLOW_FRAME_MS = 22;
const FAST_FRAME_MS = 15;
const SLOW_FRAME_LIMIT = 8;
const FAST_SETTLING_UPGRADE_FRAMES = 12;
const FAST_IDLE_UPGRADE_FRAMES = 45;
const MOTION_RECENT_INPUT_SECONDS = 0.09;
const SCENE_UNSTABLE_SECONDS = 0.6;

const QUALITY_PROFILES: Record<RenderQualityMode, QualityProfile> = {
  motion: {
    scale: 0.38,
    maxDpr: 1.0,
    minWidth: 160,
    minHeight: 96,
    stepBudget: 520,
    starLayerCount: 1,
    stepScale: 1.75,
    accumulationBlend: 0,
    modeFlag: 0
  },
  settling: {
    scale: 0.52,
    maxDpr: 1.15,
    minWidth: 208,
    minHeight: 120,
    stepBudget: 900,
    starLayerCount: 2,
    stepScale: 1.18,
    accumulationBlend: 0,
    modeFlag: 1
  },
  idle: {
    scale: 0.68,
    maxDpr: 1.25,
    minWidth: 272,
    minHeight: 160,
    stepBudget: 1400,
    starLayerCount: 3,
    stepScale: 0.92,
    accumulationBlend: 0.18,
    modeFlag: 2
  }
};

function qualityRank(mode: RenderQualityMode): number {
  switch (mode) {
    case "motion":
      return 0;
    case "settling":
      return 1;
    case "idle":
      return 2;
  }
}

class PerformanceController {
  private mode: RenderQualityMode = "settling";
  private emaFrameMs = 16.67;
  private slowFrameCount = 0;
  private fastFrameCount = 0;
  private unstableUntil = 0;
  private lastGravityEnabled = false;

  beginFrame(frameTimeMs: number, simulation: Simulation): RenderQualityMode {
    if (simulation.gravityEnabled !== this.lastGravityEnabled) {
      this.lastGravityEnabled = simulation.gravityEnabled;
      this.markSceneUnstable(simulation.time);
      this.mode = "motion";
      return this.mode;
    }

    this.emaFrameMs = this.emaFrameMs * (1.0 - EMA_ALPHA) + frameTimeMs * EMA_ALPHA;

    const recentInput = simulation.time - simulation.camera.lastInteractionTime < MOTION_RECENT_INPUT_SECONDS;
    if (simulation.camera.dragging || recentInput) {
      this.fastFrameCount = 0;
      this.slowFrameCount = 0;
      this.mode = "motion";
      return this.mode;
    }

    if (simulation.camera.isInteracting || simulation.time < this.unstableUntil) {
      this.fastFrameCount = 0;
      this.slowFrameCount = 0;
      this.mode = "settling";
      return this.mode;
    }

    if (this.emaFrameMs > SLOW_FRAME_MS) {
      this.slowFrameCount += 1;
      this.fastFrameCount = 0;
    } else if (this.emaFrameMs < FAST_FRAME_MS) {
      this.fastFrameCount += 1;
      this.slowFrameCount = 0;
    } else {
      this.fastFrameCount = 0;
      this.slowFrameCount = 0;
    }

    if (this.slowFrameCount >= SLOW_FRAME_LIMIT) {
      this.slowFrameCount = 0;
      this.fastFrameCount = 0;
      this.mode = this.mode === "idle" ? "settling" : "motion";
      this.unstableUntil = simulation.time + SCENE_UNSTABLE_SECONDS;
      return this.mode;
    }

    if (this.mode === "motion" && this.fastFrameCount >= FAST_SETTLING_UPGRADE_FRAMES) {
      this.mode = "settling";
      this.fastFrameCount = 0;
      return this.mode;
    }

    if (this.mode === "settling" && this.fastFrameCount >= FAST_IDLE_UPGRADE_FRAMES) {
      this.mode = "idle";
      this.fastFrameCount = 0;
    }

    return this.mode;
  }

  currentQuality(): RenderQualityMode {
    return this.mode;
  }

  markSceneUnstable(time: number): void {
    this.unstableUntil = Math.max(this.unstableUntil, time + SCENE_UNSTABLE_SECONDS);
    this.fastFrameCount = 0;
    this.slowFrameCount = 0;
    if (this.mode === "idle") {
      this.mode = "settling";
    }
  }
}

export class WebGPURenderer {
  private readonly context: GPUCanvasContext;
  private readonly performanceController = new PerformanceController();
  private sampler!: GPUSampler;
  private device!: GPUDevice;
  private format!: GPUTextureFormat;
  private computePipeline!: GPUComputePipeline;
  private fullscreenPipeline!: GPURenderPipeline;
  private gridPipeline!: GPURenderPipeline;
  private cameraBuffer!: GPUBuffer;
  private sceneBuffer!: GPUBuffer;
  private objectBuffer!: GPUBuffer;
  private gridUniformBuffer!: GPUBuffer;
  private gridVertexBuffer!: GPUBuffer;
  private gridIndexBuffer!: GPUBuffer;
  private computeBindGroup!: GPUBindGroup;
  private fullscreenBindGroup!: GPUBindGroup;
  private gridBindGroup!: GPUBindGroup;
  private renderTexture!: GPUTexture;
  private renderTextureView!: GPUTextureView;
  private historyTexture!: GPUTexture;
  private historyTextureView!: GPUTextureView;
  private renderWidth = 1;
  private renderHeight = 1;
  private gridIndexCount = 0;
  private presentationSize = { width: 1, height: 1 };
  private qualityMode: RenderQualityMode = "settling";
  private historyValid = false;

  constructor(private readonly canvas: HTMLCanvasElement) {
    const context = canvas.getContext("webgpu");
    if (!context) {
      throw new Error("WebGPU canvas context is unavailable.");
    }
    this.context = context;
  }

  async init(simulation: Simulation): Promise<void> {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
      throw new Error("No WebGPU adapter found.");
    }

    this.device = await adapter.requestDevice();
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: "opaque"
    });

    this.sampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear"
    });

    this.cameraBuffer = this.device.createBuffer({
      size: 5 * 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.sceneBuffer = this.device.createBuffer({
      size: 3 * 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.objectBuffer = this.device.createBuffer({
      size: MAX_OBJECTS * 3 * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    this.gridUniformBuffer = this.device.createBuffer({
      size: 16 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    const gridVertices = simulation.consumeGridVertices() ?? simulation.getGridVertices();
    const gridIndices = simulation.getGridIndices();
    this.gridVertexBuffer = this.device.createBuffer({
      size: gridVertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    });
    this.gridIndexBuffer = this.device.createBuffer({
      size: gridIndices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
    });
    this.writeTypedArray(this.gridVertexBuffer, gridVertices);
    this.writeTypedArray(this.gridIndexBuffer, gridIndices);
    this.gridIndexCount = gridIndices.length;

    const objectData = simulation.consumeObjectData();
    if (objectData) {
      this.writeTypedArray(this.objectBuffer, objectData);
    }

    const computeModule = this.device.createShaderModule({ code: computeWGSL });
    const fullscreenModule = this.device.createShaderModule({ code: fullscreenWGSL });
    const gridModule = this.device.createShaderModule({ code: gridWGSL });

    this.computePipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: computeModule,
        entryPoint: "main"
      }
    });

    this.fullscreenPipeline = this.device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: fullscreenModule,
        entryPoint: "vsMain"
      },
      fragment: {
        module: fullscreenModule,
        entryPoint: "fsMain",
        targets: [{ format: this.format }]
      },
      primitive: {
        topology: "triangle-list"
      }
    });

    this.gridPipeline = this.device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: gridModule,
        entryPoint: "vsMain",
        buffers: [
          {
            arrayStride: 12,
            attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }]
          }
        ]
      },
      fragment: {
        module: gridModule,
        entryPoint: "fsMain",
        targets: [
          {
            format: this.format,
            blend: {
              color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" }
            }
          }
        ]
      },
      primitive: {
        topology: "line-list"
      }
    });

    this.gridBindGroup = this.device.createBindGroup({
      layout: this.gridPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.gridUniformBuffer } }]
    });

    this.resizeIfNeeded(true, this.performanceController.currentQuality());
  }

  beginFrame(frameTimeMs: number, simulation: Simulation): void {
    const previousMode = this.performanceController.currentQuality();
    const nextMode = this.performanceController.beginFrame(frameTimeMs, simulation);

    if (qualityRank(nextMode) < qualityRank(previousMode)) {
      this.historyValid = false;
    }

    if (nextMode !== "idle" || simulation.gravityEnabled) {
      this.historyValid = false;
    }
  }

  currentQuality(): RenderQualityMode {
    return this.performanceController.currentQuality();
  }

  render(simulation: Simulation): void {
    const qualityMode = this.performanceController.currentQuality();
    const profile = QUALITY_PROFILES[qualityMode];
    this.resizeIfNeeded(false, qualityMode);

    const frame = simulation.camera.frame();
    const aspect = this.renderWidth / Math.max(1, this.renderHeight);
    const moving = simulation.camera.isInteracting ? 1 : 0;
    const accumulationBlend = qualityMode === "idle" && this.historyValid && !simulation.gravityEnabled
      ? profile.accumulationBlend
      : 0;

    const cameraData = new Float32Array([
      frame.position.x, frame.position.y, frame.position.z, 0,
      frame.right.x, frame.right.y, frame.right.z, 0,
      frame.up.x, frame.up.y, frame.up.z, 0,
      frame.forward.x, frame.forward.y, frame.forward.z, 0,
      Math.tan((60 * Math.PI) / 360), aspect, simulation.time, moving
    ]);
    this.writeTypedArray(this.cameraBuffer, cameraData);

    const sceneData = new Float32Array([
      BLACK_HOLE_RS * 2.2,
      BLACK_HOLE_RS * 5.2,
      BLACK_HOLE_RS,
      simulation.objects.length,
      profile.stepBudget,
      profile.starLayerCount,
      profile.stepScale,
      accumulationBlend,
      profile.modeFlag,
      0,
      0,
      0
    ]);
    this.writeTypedArray(this.sceneBuffer, sceneData);

    const objectData = simulation.consumeObjectData();
    if (objectData) {
      this.writeTypedArray(this.objectBuffer, objectData);
    }

    const gridVertices = simulation.consumeGridVertices();
    if (gridVertices) {
      this.writeTypedArray(this.gridVertexBuffer, gridVertices);
    }

    this.writeTypedArray(
      this.gridUniformBuffer,
      simulation.viewProjection(this.presentationSize.width / Math.max(1, this.presentationSize.height))
    );

    const commandEncoder = this.device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.computePipeline);
    computePass.setBindGroup(0, this.computeBindGroup);
    computePass.dispatchWorkgroups(Math.ceil(this.renderWidth / 8), Math.ceil(this.renderHeight / 8));
    computePass.end();

    const shouldAccumulate = qualityMode === "idle" && !simulation.gravityEnabled;
    if (shouldAccumulate) {
      commandEncoder.copyTextureToTexture(
        { texture: this.renderTexture },
        { texture: this.historyTexture },
        { width: this.renderWidth, height: this.renderHeight, depthOrArrayLayers: 1 }
      );
    }

    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0.01, g: 0.015, b: 0.03, a: 1 },
          loadOp: "clear",
          storeOp: "store"
        }
      ]
    });
    renderPass.setPipeline(this.fullscreenPipeline);
    renderPass.setBindGroup(0, this.fullscreenBindGroup);
    renderPass.draw(6);
    renderPass.setPipeline(this.gridPipeline);
    renderPass.setBindGroup(0, this.gridBindGroup);
    renderPass.setVertexBuffer(0, this.gridVertexBuffer);
    renderPass.setIndexBuffer(this.gridIndexBuffer, "uint32");
    renderPass.drawIndexed(this.gridIndexCount);
    renderPass.end();

    this.device.queue.submit([commandEncoder.finish()]);
    this.historyValid = shouldAccumulate;
  }

  private resizeIfNeeded(force: boolean, qualityMode: RenderQualityMode): void {
    const profile = QUALITY_PROFILES[qualityMode];
    const devicePixelRatio = Math.min(window.devicePixelRatio || 1, profile.maxDpr);
    const nextWidth = Math.max(1, Math.floor(this.canvas.clientWidth * devicePixelRatio));
    const nextHeight = Math.max(1, Math.floor(this.canvas.clientHeight * devicePixelRatio));
    const desiredRenderWidth = Math.max(profile.minWidth, Math.floor(nextWidth * profile.scale));
    const desiredRenderHeight = Math.max(profile.minHeight, Math.floor(nextHeight * profile.scale));

    const presentationChanged =
      nextWidth !== this.presentationSize.width || nextHeight !== this.presentationSize.height;
    const renderChanged =
      desiredRenderWidth !== this.renderWidth || desiredRenderHeight !== this.renderHeight;
    const qualityChanged = qualityMode !== this.qualityMode;

    if (!force && !presentationChanged && !renderChanged && !qualityChanged) {
      return;
    }

    this.presentationSize = { width: nextWidth, height: nextHeight };
    this.qualityMode = qualityMode;
    this.canvas.width = nextWidth;
    this.canvas.height = nextHeight;
    this.renderWidth = desiredRenderWidth;
    this.renderHeight = desiredRenderHeight;
    this.historyValid = false;

    if (this.renderTexture) {
      this.renderTexture.destroy();
    }
    if (this.historyTexture) {
      this.historyTexture.destroy();
    }

    this.renderTexture = this.device.createTexture({
      size: { width: this.renderWidth, height: this.renderHeight },
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC
    });
    this.renderTextureView = this.renderTexture.createView();

    this.historyTexture = this.device.createTexture({
      size: { width: this.renderWidth, height: this.renderHeight },
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
    });
    this.historyTextureView = this.historyTexture.createView();

    this.computeBindGroup = this.device.createBindGroup({
      layout: this.computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.cameraBuffer } },
        { binding: 1, resource: { buffer: this.sceneBuffer } },
        { binding: 2, resource: { buffer: this.objectBuffer } },
        { binding: 3, resource: this.renderTextureView },
        { binding: 4, resource: this.historyTextureView }
      ]
    });

    this.fullscreenBindGroup = this.device.createBindGroup({
      layout: this.fullscreenPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: this.renderTextureView }
      ]
    });
  }

  private writeTypedArray(buffer: GPUBuffer, data: Float32Array | Uint32Array): void {
    const source = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const bytes = new Uint8Array(data.byteLength);
    bytes.set(source);
    this.device.queue.writeBuffer(buffer, 0, bytes);
  }
}
