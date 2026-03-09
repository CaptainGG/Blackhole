export class AudioController {
  private context: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private lowGain: GainNode | null = null;
  private midGain: GainNode | null = null;
  private highGain: GainNode | null = null;
  private noiseGain: GainNode | null = null;
  private highOsc: OscillatorNode | null = null;
  private activated = false;

  async activate(): Promise<void> {
    if (this.activated) {
      if (this.context?.state === "suspended") {
        await this.context.resume();
      }
      return;
    }

    this.context = new AudioContext();
    this.masterGain = this.context.createGain();
    this.masterGain.gain.value = 0;
    this.masterGain.connect(this.context.destination);

    const low = this.context.createOscillator();
    low.type = "sine";
    low.frequency.value = 30;
    this.lowGain = this.context.createGain();
    this.lowGain.gain.value = 0.06;
    low.connect(this.lowGain).connect(this.masterGain);
    low.start();

    const mid = this.context.createOscillator();
    mid.type = "sine";
    mid.frequency.value = 55;
    this.midGain = this.context.createGain();
    this.midGain.gain.value = 0.04;
    mid.connect(this.midGain).connect(this.masterGain);
    mid.start();

    this.highOsc = this.context.createOscillator();
    this.highOsc.type = "sine";
    this.highOsc.frequency.value = 80;
    this.highGain = this.context.createGain();
    this.highGain.gain.value = 0.025;
    this.highOsc.connect(this.highGain).connect(this.masterGain);
    this.highOsc.start();

    const noiseBuffer = this.context.createBuffer(1, this.context.sampleRate * 2, this.context.sampleRate);
    const channel = noiseBuffer.getChannelData(0);
    for (let i = 0; i < channel.length; i += 1) {
      channel[i] = Math.random() * 2 - 1;
    }
    const noise = this.context.createBufferSource();
    noise.buffer = noiseBuffer;
    noise.loop = true;
    const lowPass = this.context.createBiquadFilter();
    lowPass.type = "lowpass";
    lowPass.frequency.value = 280;
    this.noiseGain = this.context.createGain();
    this.noiseGain.gain.value = 0.015;
    noise.connect(lowPass).connect(this.noiseGain).connect(this.masterGain);
    noise.start();

    await this.context.resume();
    this.activated = true;
  }

  update(gravBoost: number): void {
    if (!this.context || !this.masterGain) {
      return;
    }

    const time = this.context.currentTime;
    const base = 0.05 + gravBoost * 0.2;
    this.masterGain.gain.linearRampToValueAtTime(base, time + 0.08);
    this.lowGain?.gain.linearRampToValueAtTime(0.05 + gravBoost * 0.06, time + 0.08);
    this.midGain?.gain.linearRampToValueAtTime(0.035 + gravBoost * 0.05, time + 0.08);
    this.highGain?.gain.linearRampToValueAtTime(0.018 + gravBoost * 0.04, time + 0.08);
    this.noiseGain?.gain.linearRampToValueAtTime(0.01 + gravBoost * 0.05, time + 0.08);
    this.highOsc?.frequency.linearRampToValueAtTime(80 + gravBoost * 16, time + 0.08);
  }
}
