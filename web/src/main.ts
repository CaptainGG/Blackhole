import "./styles.css";

import { AudioController } from "./audio";
import { InputController } from "./input-controller";
import { WebGPURenderer } from "./renderer";
import { Simulation } from "./simulation";

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) {
    throw new Error(`Missing required element: ${selector}`);
  }
  return element;
}

const app = requireElement<HTMLDivElement>("#app");
const canvas = requireElement<HTMLCanvasElement>("#scene");
const status = requireElement<HTMLParagraphElement>("#status");
const copyButton = requireElement<HTMLButtonElement>("#copy-link");
const unsupported = requireElement<HTMLDivElement>("#unsupported");

let currentStatus = "";
let statusResetHandle: number | null = null;
let resetStatusToDefault = () => {};

function setStatus(message: string): void {
  if (message === currentStatus) {
    return;
  }
  currentStatus = message;
  status.textContent = message;
}

function flashStatus(message: string, duration = 1800): void {
  if (statusResetHandle !== null) {
    window.clearTimeout(statusResetHandle);
  }
  setStatus(message);
  statusResetHandle = window.setTimeout(() => {
    statusResetHandle = null;
    resetStatusToDefault();
  }, duration);
}

copyButton.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(window.location.href);
    flashStatus("Link copied.");
  } catch {
    flashStatus("Clipboard unavailable in this browser.", 2200);
  }
});

if (!("gpu" in navigator)) {
  unsupported.classList.remove("hidden");
  setStatus("WebGPU unavailable.");
} else {
  void bootstrap();
}

async function bootstrap(): Promise<void> {
  const simulation = new Simulation();
  const audio = new AudioController();
  const renderer = new WebGPURenderer(canvas);

  let audioUnlocked = false;
  let previousGravityEnabled = simulation.gravityEnabled;
  let previousInteractionState = false;

  const applyDefaultStatus = () => {
    if (!audioUnlocked) {
      setStatus("WebGPU active. Click or drag to unlock audio.");
      return;
    }

    if (simulation.gravityEnabled) {
      setStatus("Gravity engaged. Drag to inspect the bend.");
      return;
    }

    setStatus("Drag to orbit. Wheel to dive in. Press G for gravity.");
  };

  resetStatusToDefault = applyDefaultStatus;

  const input = new InputController(canvas, simulation, () => {
    if (audioUnlocked) {
      void audio.activate();
      return;
    }

    void audio.activate().then(() => {
      audioUnlocked = true;
      applyDefaultStatus();
    });
  });

  try {
    await renderer.init(simulation);
    input.attach();
    applyDefaultStatus();
  } catch (error) {
    unsupported.classList.remove("hidden");
    setStatus(error instanceof Error ? error.message : "WebGPU initialization failed.");
    return;
  }

  let last = performance.now();
  const frame = (now: number) => {
    const frameTimeMs = now - last;
    const dt = Math.min(frameTimeMs / 1000, 1 / 20);
    last = now;

    simulation.update(dt);
    renderer.beginFrame(frameTimeMs, simulation);
    audio.update(simulation.gravBoost);
    renderer.render(simulation);

    if (simulation.gravityEnabled !== previousGravityEnabled) {
      previousGravityEnabled = simulation.gravityEnabled;
      flashStatus(
        simulation.gravityEnabled ? "Gravity simulation engaged." : "Gravity simulation paused.",
        1400
      );
    }

    if (simulation.camera.isInteracting !== previousInteractionState) {
      previousInteractionState = simulation.camera.isInteracting;
      app.classList.toggle("is-interacting", previousInteractionState);
    }

    requestAnimationFrame(frame);
  };

  requestAnimationFrame(frame);
}
