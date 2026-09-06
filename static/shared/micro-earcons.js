/**
 * static/shared/micro-earcons.js
 *
 * Zero-dependency, procedural auditory feedback synthesizer using Web Audio API oscillators.
 * Provides subtle non-intrusive earcons for:
 *  - startListening: User session or microphone active
 *  - expertDelegation: Delegation initiated to a frontier model or tool
 *  - voiceReturn: Synthesis of expert response returning to duplex voice
 *  - bargeIn: Soft tick confirming assistant stopped speaking when interrupted
 */

export class MicroEarcons {
    /**
     * @param {AudioContext} [sharedCtx] - Optional shared AudioContext
     */
    constructor(sharedCtx = null) {
        this._ctx = sharedCtx;
        this.enabled = true;
        this.volume = 0.25; // Subtle, non-intrusive default
    }

    _ensureCtx() {
        if (!this._ctx || this._ctx.state === 'closed') {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (AudioCtx) {
                this._ctx = new AudioCtx();
            }
        }
        if (this._ctx && this._ctx.state === 'suspended') {
            this._ctx.resume().catch(() => {});
        }
        return this._ctx;
    }

    setAudioContext(ctx) {
        this._ctx = ctx;
    }

    /**
     * Subtle two-tone rising chime confirming listening state.
     */
    playStartListening() {
        if (!this.enabled) return;
        const ctx = this._ensureCtx();
        if (!ctx) return;

        const now = ctx.currentTime;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();

        osc.type = 'sine';
        osc.frequency.setValueAtTime(523.25, now); // C5
        osc.frequency.exponentialRampToValueAtTime(659.25, now + 0.08); // E5

        gain.gain.setValueAtTime(0.0001, now);
        gain.gain.linearRampToValueAtTime(0.12 * this.volume, now + 0.015);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.12);

        osc.connect(gain);
        gain.connect(ctx.destination);

        osc.start(now);
        osc.stop(now + 0.13);
    }

    /**
     * Dual-harmonic bell ping notifying background expert delegation or tool call.
     */
    playExpertDelegation() {
        if (!this.enabled) return;
        const ctx = this._ensureCtx();
        if (!ctx) return;

        const now = ctx.currentTime;
        [880, 1320].forEach((freq, idx) => {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();

            osc.type = 'sine';
            osc.frequency.setValueAtTime(freq, now);

            gain.gain.setValueAtTime(0.0001, now);
            const peak = (idx === 0 ? 0.15 : 0.08) * this.volume;
            gain.gain.linearRampToValueAtTime(peak, now + 0.01);
            gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.22);

            osc.connect(gain);
            gain.connect(ctx.destination);

            osc.start(now);
            osc.stop(now + 0.23);
        });
    }

    /**
     * Upward soft chime sweep notifying expert answer returned to duplex voice.
     */
    playVoiceReturn() {
        if (!this.enabled) return;
        const ctx = this._ensureCtx();
        if (!ctx) return;

        const now = ctx.currentTime;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();

        osc.type = 'sine';
        osc.frequency.setValueAtTime(440, now);
        osc.frequency.exponentialRampToValueAtTime(880, now + 0.09);

        gain.gain.setValueAtTime(0.0001, now);
        gain.gain.linearRampToValueAtTime(0.14 * this.volume, now + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.18);

        osc.connect(gain);
        gain.connect(ctx.destination);

        osc.start(now);
        osc.stop(now + 0.19);
    }

    /**
     * Damped soft click/chirp acknowledging immediate turn stop on user barge-in.
     */
    playBargeIn() {
        if (!this.enabled) return;
        const ctx = this._ensureCtx();
        if (!ctx) return;

        const now = ctx.currentTime;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();

        osc.type = 'triangle';
        osc.frequency.setValueAtTime(360, now);
        osc.frequency.exponentialRampToValueAtTime(180, now + 0.04);

        gain.gain.setValueAtTime(0.0001, now);
        gain.gain.linearRampToValueAtTime(0.08 * this.volume, now + 0.005);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.05);

        osc.connect(gain);
        gain.connect(ctx.destination);

        osc.start(now);
        osc.stop(now + 0.06);
    }
}
