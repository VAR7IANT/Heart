/*
 * Heart rhythm + synthesized heartbeat sound.
 *
 * The rhythm is the single source of truth for both the visuals and the
 * audio: beat onsets are a deterministic sequence (with gentle, natural
 * heart-rate variability), the renderer reads "time since onset" from it,
 * and the audio engine schedules each lub-dub on the Web Audio clock ahead of
 * time. Nothing is streamed, so playback works offline and never drifts.
 */
(() => {
  'use strict';

  const BPM = 62;
  const DUB_DELAY = 0.30; // S1 → S2 interval (seconds) at rest.

  /** Deterministic beat onset sequence with respiratory-like variability. */
  class HeartRhythm {
    constructor(bpm = BPM) {
      this.base = 60 / bpm;
      this.index = 0;
      this.current = 0;               // onset of the beat in progress
      this.previous = -this.base;     // onset of the one before it
      this.next = this.period(0);
    }

    period(k) {
      // Slow sinus-arrhythmia swing plus a small, smooth irregularity.
      const breath = Math.sin(k * 0.86) * 0.035;
      const drift = Math.sin(k * 2.17 + 1.3) * 0.012 + Math.sin(k * 0.31) * 0.02;
      return this.base * (1 + breath + drift);
    }

    /** Advance so that current <= t < next. Time only moves forward. */
    update(t) {
      while (t >= this.next) {
        this.previous = this.current;
        this.current = this.next;
        this.index++;
        this.next = this.current + this.period(this.index);
      }
    }

    /** Onsets of upcoming beats, starting with the one in progress. */
    upcoming(count) {
      const list = [{ index: this.index, time: this.current }];
      let t = this.next;
      for (let k = this.index + 1; list.length < count; k++) {
        list.push({ index: k, time: t });
        t += this.period(k);
      }
      return list;
    }
  }

  /** Smooth lub-dub envelope (C1 continuous, 0 at onset, peak 1). */
  function pulse(x, tau) {
    if (x <= 0) return 0;
    const u = x / (2 * tau);
    return u * u * Math.exp(2 - x / tau);
  }
  function beatShape(age) {
    return pulse(age, 0.045) + 0.56 * pulse(age - DUB_DELAY, 0.04);
  }

  class HeartbeatAudio {
    constructor() {
      this.ctx = null;
      this.master = null;
      this.muted = false;
      this.active = true;
      this.scheduled = new Set();
      this.noise = null;
    }

    get running() {
      return !!this.ctx && this.ctx.state === 'running';
    }

    build() {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (!AudioContext) return false;
      const ctx = new AudioContext({ latencyHint: 'interactive' });
      const master = ctx.createGain();
      master.gain.value = 0;
      // Gentle chest-like resonance and protection against clipping.
      const body = ctx.createBiquadFilter();
      body.type = 'lowpass';
      body.frequency.value = 420;
      body.Q.value = 0.7;
      const comp = ctx.createDynamicsCompressor();
      comp.threshold.value = -14;
      comp.knee.value = 10;
      comp.ratio.value = 3;
      comp.attack.value = 0.003;
      comp.release.value = 0.2;
      body.connect(comp).connect(master).connect(ctx.destination);

      // Short buffer of brown noise for the valve "texture".
      const length = Math.floor(ctx.sampleRate * 0.4);
      const buffer = ctx.createBuffer(1, length, ctx.sampleRate);
      const data = buffer.getChannelData(0);
      let last = 0;
      for (let i = 0; i < length; i++) {
        last = (last + 0.02 * (Math.random() * 2 - 1)) / 1.02;
        data[i] = last * 3.5;
      }
      this.noise = buffer;
      this.ctx = ctx;
      this.master = master;
      this.input = body;
      return true;
    }

    /** One heart sound: a pitched thump with a soft noise transient. */
    thump(at, { from, to, decay, gain }) {
      const ctx = this.ctx;
      const osc = ctx.createOscillator();
      const env = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(from, at);
      osc.frequency.exponentialRampToValueAtTime(to, at + decay * 0.9);
      env.gain.setValueAtTime(0.0001, at);
      env.gain.exponentialRampToValueAtTime(gain, at + 0.012);
      env.gain.exponentialRampToValueAtTime(0.0001, at + decay);
      osc.connect(env).connect(this.input);
      osc.start(at);
      osc.stop(at + decay + 0.05);

      // Second harmonic keeps the beat audible on small phone speakers.
      const over = ctx.createOscillator();
      const overEnv = ctx.createGain();
      over.type = 'sine';
      over.frequency.setValueAtTime(from * 2.1, at);
      over.frequency.exponentialRampToValueAtTime(to * 2, at + decay * 0.6);
      overEnv.gain.setValueAtTime(0.0001, at);
      overEnv.gain.exponentialRampToValueAtTime(gain * 0.32, at + 0.008);
      overEnv.gain.exponentialRampToValueAtTime(0.0001, at + decay * 0.55);
      over.connect(overEnv).connect(this.input);
      over.start(at);
      over.stop(at + decay);

      const src = ctx.createBufferSource();
      const band = ctx.createBiquadFilter();
      const noiseEnv = ctx.createGain();
      src.buffer = this.noise;
      band.type = 'bandpass';
      band.frequency.value = 150;
      band.Q.value = 0.9;
      noiseEnv.gain.setValueAtTime(0.0001, at);
      noiseEnv.gain.exponentialRampToValueAtTime(gain * 0.5, at + 0.006);
      noiseEnv.gain.exponentialRampToValueAtTime(0.0001, at + decay * 0.7);
      src.connect(band).connect(noiseEnv).connect(this.input);
      src.start(at);
      src.stop(at + decay);
    }

    /**
     * Schedule any beats whose sound must start soon. `visualNow` is the
     * renderer clock; beats are mapped onto the audio clock so the sound
     * reaches the ear at the same moment the heart contracts on screen.
     */
    schedule(rhythm, visualNow) {
      if (!this.running || this.muted || !this.active) return;
      const ctx = this.ctx;
      const latency = (ctx.outputLatency || 0) + (ctx.baseLatency || 0);
      const offset = ctx.currentTime - visualNow - latency;
      for (const beat of rhythm.upcoming(3)) {
        if (this.scheduled.has(beat.index)) continue;
        const at = beat.time + offset;
        if (at > ctx.currentTime + 0.35) break;
        this.scheduled.add(beat.index);
        // Don't play beats that are already audibly late.
        if (at < ctx.currentTime - 0.03) continue;
        const t = Math.max(at, ctx.currentTime + 0.005);
        this.thump(t, { from: 78, to: 44, decay: 0.2, gain: 0.9 });
        this.thump(t + DUB_DELAY, { from: 98, to: 56, decay: 0.14, gain: 0.62 });
      }
      if (this.scheduled.size > 16) {
        const keep = rhythm.index - 2;
        for (const i of this.scheduled) if (i < keep) this.scheduled.delete(i);
      }
    }

    async enable() {
      if (!this.ctx && !this.build()) return;
      try {
        if (this.ctx.state !== 'running') await this.ctx.resume();
      } catch { /* Autoplay policy: the next gesture retries. */ }
      this.applyGain();
    }

    applyGain() {
      if (!this.ctx) return;
      const g = this.master.gain, now = this.ctx.currentTime;
      g.cancelScheduledValues(now);
      g.setValueAtTime(g.value, now);
      g.linearRampToValueAtTime(this.muted || !this.active ? 0 : 0.85, now + 0.4);
    }

    toggleMuted() {
      this.muted = !this.muted;
      if (!this.muted && !this.ctx) return this.enable();
      this.applyGain();
    }

    setActive(active) {
      this.active = active;
      if (!this.ctx) return;
      this.applyGain();
      if (active) this.ctx.resume().catch(() => {});
      else setTimeout(() => { if (!this.active && this.ctx) this.ctx.suspend().catch(() => {}); }, 450);
    }

    dispose() {
      if (this.ctx) this.ctx.close().catch(() => {});
      this.ctx = null;
    }
  }

  window.Heartbeat = { HeartRhythm, HeartbeatAudio, beatShape, DUB_DELAY };
})();
