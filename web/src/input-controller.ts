import type { Simulation } from "./simulation";

const LINE_HEIGHT_PIXELS = 16;
const PAGE_HEIGHT_PIXELS = 120;

export class InputController {
  private readonly onActivateAudio: () => void;
  private activePointerId: number | null = null;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly simulation: Simulation,
    onActivateAudio: () => void
  ) {
    this.onActivateAudio = onActivateAudio;
  }

  attach(): void {
    this.canvas.addEventListener("pointerdown", this.handlePointerDown);
    this.canvas.addEventListener("pointermove", this.handlePointerMove);
    this.canvas.addEventListener("pointerup", this.handlePointerUp);
    this.canvas.addEventListener("pointercancel", this.handlePointerCancel);
    this.canvas.addEventListener("wheel", this.handleWheel, { passive: false });
    window.addEventListener("keydown", this.handleKeyDown);
  }

  dispose(): void {
    this.canvas.removeEventListener("pointerdown", this.handlePointerDown);
    this.canvas.removeEventListener("pointermove", this.handlePointerMove);
    this.canvas.removeEventListener("pointerup", this.handlePointerUp);
    this.canvas.removeEventListener("pointercancel", this.handlePointerCancel);
    this.canvas.removeEventListener("wheel", this.handleWheel);
    window.removeEventListener("keydown", this.handleKeyDown);
  }

  private handlePointerDown = (event: PointerEvent): void => {
    if (event.button !== 0 || this.activePointerId !== null) {
      return;
    }

    event.preventDefault();
    this.onActivateAudio();
    this.activePointerId = event.pointerId;
    this.canvas.setPointerCapture(event.pointerId);
    this.simulation.camera.beginDrag(event.clientX, event.clientY, this.simulation.time);
  };

  private handlePointerMove = (event: PointerEvent): void => {
    if (event.pointerId !== this.activePointerId) {
      return;
    }

    this.simulation.camera.pointerMove(
      event.clientX,
      event.clientY,
      this.canvas.clientWidth,
      this.canvas.clientHeight,
      this.simulation.time
    );
  };

  private handlePointerUp = (event: PointerEvent): void => {
    if (event.pointerId !== this.activePointerId) {
      return;
    }

    this.endPointerInteraction(event.pointerId);
  };

  private handlePointerCancel = (event: PointerEvent): void => {
    if (event.pointerId !== this.activePointerId) {
      return;
    }

    this.endPointerInteraction(event.pointerId);
  };

  private handleWheel = (event: WheelEvent): void => {
    event.preventDefault();
    this.onActivateAudio();
    const delta = this.normalizeWheelDelta(event);
    const limitedDelta = Math.max(-160, Math.min(160, delta));
    this.simulation.camera.zoom(limitedDelta, this.simulation.time);
  };

  private handleKeyDown = (event: KeyboardEvent): void => {
    if (event.key.toLowerCase() !== "g") {
      return;
    }

    this.simulation.toggleGravity();
    this.onActivateAudio();
  };

  private endPointerInteraction(pointerId: number): void {
    if (this.canvas.hasPointerCapture(pointerId)) {
      this.canvas.releasePointerCapture(pointerId);
    }
    this.activePointerId = null;
    this.simulation.camera.endDrag(this.simulation.time);
  }

  private normalizeWheelDelta(event: WheelEvent): number {
    switch (event.deltaMode) {
      case WheelEvent.DOM_DELTA_LINE:
        return event.deltaY * LINE_HEIGHT_PIXELS;
      case WheelEvent.DOM_DELTA_PAGE:
        return event.deltaY * PAGE_HEIGHT_PIXELS;
      default:
        return event.deltaY;
    }
  }
}