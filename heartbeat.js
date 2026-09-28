(() => {
  'use strict';
  const wrap = value => ((value % 1) + 1) % 1;

  class HeartbeatAudio {
    constructor() {
      this.period = 1 / .76;
      this.context = null;
      this.source = null;
      this.master = null;
      this.epoch = 0;
      this.active = true;
      this.muted = false;
      this.starting = null;
      this.disposed = false;
    }

    createBuffer() {
      const rate = this.context.sampleRate;
      const length = Math.round(this.period * rate);
      this.period = length / rate;
      const buffer = this.context.createBuffer(1, length, rate);
      const samples = buffer.getChannelData(0);
      let noise = 0;
      for (let i = 0; i < length; i++) {
        const seconds = i / rate;
        noise = noise * .91 + (Math.random() * 2 - 1) * .09;
        for (const [phase, strength, frequency] of [[.12, .48, 54], [.32, .3, 68]]) {
          const relative = seconds - phase * this.period;
          if (relative < -.065 || relative > .27) continue;
          const local = relative + .065;
          const envelope = Math.exp(-((relative / (relative < 0 ? .023 : .076)) ** 2));
          const angle = Math.PI * 2 * (frequency * local + 32 * .04 * (1 - Math.exp(-local / .04)));
          const body = Math.sin(angle) * .77 + Math.sin(angle * 2.04) * .16 + noise * .2;
          samples[i] += body * envelope * strength;
        }
      }
      return buffer;
    }

    // The audible device clock also drives the animation, so frames cannot
    // gradually drift away from a separate sound loop or timer.
    outputTime() {
      const context = this.context;
      let output = context.currentTime - (context.outputLatency || 0) - (context.baseLatency || 0);
      if (context.state === 'running' && context.getOutputTimestamp) {
        const stamp = context.getOutputTimestamp();
        if (stamp.contextTime > 0 && stamp.performanceTime > 0) {
          output = stamp.contextTime + Math.max(0, performance.now() - stamp.performanceTime) / 1000;
        }
      }
      return Math.max(0, Math.min(context.currentTime, output));
    }

    phase(visualTime) {
      if (this.source && this.context.state === 'running') return wrap((this.outputTime() - this.epoch) / this.period);
      return wrap(visualTime / this.period);
    }

    async enable(visualTime) {
      if (this.disposed || !this.active) return;
      if (this.starting) return this.starting;
      const Audio = window.AudioContext || window.webkitAudioContext;
      if (!Audio) return;
      this.starting = (async () => {
        try {
          if (!this.context) this.context = new Audio({ latencyHint: 'interactive' });
          await this.context.resume();
          if (this.disposed) return;
          if (!this.active) { await this.context.suspend(); return; }
          if (!this.source) {
            const phase = wrap(visualTime / this.period);
            this.master = this.context.createGain();
            this.master.gain.setValueAtTime(0, this.context.currentTime);
            this.master.gain.linearRampToValueAtTime(this.muted ? 0 : .8, this.context.currentTime + .1);
            this.master.connect(this.context.destination);
            this.source = this.context.createBufferSource();
            this.source.buffer = this.createBuffer();
            this.source.loop = true;
            this.source.connect(this.master);
            const start = this.context.currentTime + .025;
            const offset = phase * this.period;
            this.epoch = start - offset;
            this.source.start(start, offset);
          }
        } catch {
          // Playback can be blocked or interrupted by the browser. A later
          // user gesture can retry; the visual experience keeps running.
        }
      })();
      try { await this.starting; } finally { this.starting = null; }
    }

    setActive(active) {
      this.active = active;
      if (!this.context || this.disposed) return;
      const operation = active && this.source ? this.context.resume() : this.context.suspend();
      operation.catch(() => {});
    }

    toggleMuted(visualTime) {
      if (!this.source) { void this.enable(visualTime); return; }
      this.muted = !this.muted;
      const now = this.context.currentTime;
      this.master.gain.cancelScheduledValues(now);
      this.master.gain.setTargetAtTime(this.muted ? 0 : .8, now, .025);
    }

    dispose() {
      this.disposed = true;
      if (this.source) { try { this.source.stop(); } catch {} this.source.disconnect(); }
      if (this.master) this.master.disconnect();
      if (this.context && this.context.state !== 'closed') void this.context.close().catch(() => {});
    }
  }
  window.HeartbeatAudio = HeartbeatAudio;
})();
