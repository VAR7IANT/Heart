(() => {
  'use strict';
  const wrap = value => ((value % 1) + 1) % 1;
  const HEARTBEAT_SOURCES = [
    {
      url: 'https://upload.wikimedia.org/wikipedia/commons/transcoded/b/b5/Human_heart_beating_at_61_bpm_%28Cc-by-3.0%29.ogg/Human_heart_beating_at_61_bpm_%28Cc-by-3.0%29.ogg.mp3',
      type: 'audio/mpeg'
    },
    {
      url: 'https://upload.wikimedia.org/wikipedia/commons/b/b5/Human_heart_beating_at_61_bpm_%28Cc-by-3.0%29.ogg',
      type: 'audio/ogg; codecs="vorbis"'
    }
  ];

  class HeartbeatAudio {
    constructor() {
      this.period = 60 / 61;
      this.audio = null;
      this.active = true;
      this.started = false;
      this.starting = null;
      this.disposed = false;
      this.muted = false;
      this.pendingPhase = null;
      this.loadedMetadata = false;
    }

    createAudio() {
      const audio = new Audio();
      audio.preload = 'auto';
      audio.loop = true;
      audio.volume = .48;
      audio.addEventListener('loadedmetadata', () => {
        if (Number.isFinite(audio.duration) && audio.duration > 0) {
          this.loadedMetadata = true;
          // The recording is about 14 seconds and 61 BPM. Matching the nearest
          // whole number of beats keeps the animation continuous at the loop.
          const beats = Math.max(1, Math.round(audio.duration * 61 / 60));
          this.period = audio.duration / beats;
          if (this.pendingPhase !== null) this.seekToPhase(this.pendingPhase);
        }
      });
      audio.addEventListener('playing', () => { this.started = true; });
      // MP3 covers broad browser support; Commons' original Ogg is a
      // browser-selected fallback if the MP3 rendition cannot be used.
      for (const source of HEARTBEAT_SOURCES) {
        const element = document.createElement('source');
        element.src = source.url;
        element.type = source.type;
        audio.appendChild(element);
      }
      this.audio = audio;
      return audio;
    }

    seekToPhase(phase) {
      if (!this.audio || !Number.isFinite(this.audio.duration) || !this.audio.duration) {
        this.pendingPhase = phase;
        return;
      }
      const beats = Math.max(1, Math.round(this.audio.duration * 61 / 60));
      this.period = this.audio.duration / beats;
      const target = wrap(phase) * this.period;
      if (Math.abs(this.audio.currentTime - target) > .08) {
        try { this.audio.currentTime = target; } catch { this.pendingPhase = phase; }
      }
      this.pendingPhase = null;
    }

    phase(visualTime) {
      if (this.audio && !this.audio.paused && Number.isFinite(this.audio.duration) && this.audio.duration > 0) {
        return wrap(this.audio.currentTime / this.period);
      }
      return wrap(visualTime / this.period);
    }

    async enable(visualTime) {
      if (this.disposed || !this.active) return;
      if (this.starting) return this.starting;
      const audio = this.audio || this.createAudio();
      this.starting = (async () => {
        try {
          if (audio.readyState >= HTMLMediaElement.HAVE_METADATA || this.loadedMetadata) this.seekToPhase(this.phase(visualTime));
          else this.pendingPhase = wrap(visualTime / this.period);
          audio.muted = this.muted;
          await audio.play();
          if (this.disposed || !this.active) audio.pause();
        } catch {
          // If the browser blocks playback, the animation continues and the
          // next click or key press can retry it.
        }
      })();
      try { await this.starting; } finally { this.starting = null; }
    }

    setActive(active) {
      this.active = active;
      if (!this.audio || !this.started || this.disposed) return;
      if (active) this.audio.play().catch(() => {});
      else this.audio.pause();
    }

    toggleMuted(visualTime) {
      this.muted = !this.muted;
      if (this.audio) this.audio.muted = this.muted;
      if (!this.muted && !this.started) void this.enable(visualTime);
    }

    dispose() {
      this.disposed = true;
      if (this.audio) {
        this.audio.pause();
        this.audio.removeAttribute('src');
        this.audio.replaceChildren();
        this.audio.load();
      }
    }
  }

  window.HeartbeatAudio = HeartbeatAudio;
})();
