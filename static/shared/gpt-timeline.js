export class GPTTimeline extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.shadowRoot.innerHTML = `
            <style>
                :host {
                    display: block;
                    width: 100%;
                    height: 220px;
                    background: #0f1117;
                    color: #e2e8f0;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    position: relative;
                    border-radius: 10px;
                    overflow: hidden;
                    box-sizing: border-box;
                    border: 1px solid #1e293b;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
                }
                .timeline-container {
                    display: flex;
                    width: 100%;
                    height: 100%;
                }
                .timeline-labels {
                    width: 110px;
                    height: 100%;
                    background: #131824;
                    display: flex;
                    flex-direction: column;
                    border-right: 1px solid #1e293b;
                    z-index: 10;
                    flex-shrink: 0;
                }
                .header-placeholder {
                    height: 24px;
                    border-bottom: 1px solid #1e293b;
                    display: flex;
                    align-items: center;
                    padding-left: 8px;
                    font-size: 10px;
                    color: #64748b;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }
                .label {
                    flex: 1;
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    padding-left: 10px;
                    font-size: 11px;
                    font-weight: 600;
                    letter-spacing: 0.3px;
                    border-bottom: 1px solid #1e293b;
                }
                .label:last-child {
                    border-bottom: none;
                }
                .dot {
                    width: 7px;
                    height: 7px;
                    border-radius: 50%;
                }
                .label.user { color: #34d399; }
                .label.user .dot { background: #10b981; box-shadow: 0 0 6px rgba(16, 185, 129, 0.6); }
                .label.ai { color: #60a5fa; }
                .label.ai .dot { background: #3b82f6; box-shadow: 0 0 6px rgba(59, 130, 246, 0.6); }
                .label.expert { color: #fbbf24; }
                .label.expert .dot { background: #f59e0b; box-shadow: 0 0 6px rgba(245, 158, 11, 0.6); }
                
                .timeline-scroll {
                    flex: 1;
                    height: 100%;
                    position: relative;
                    overflow: hidden;
                    cursor: crosshair;
                }
                canvas {
                    display: block;
                    width: 100%;
                    height: 100%;
                }
                .live-badge {
                    position: absolute;
                    top: 4px;
                    right: 8px;
                    background: rgba(239, 68, 68, 0.15);
                    border: 1px solid rgba(239, 68, 68, 0.5);
                    color: #ef4444;
                    font-size: 9px;
                    font-weight: 700;
                    padding: 2px 6px;
                    border-radius: 9999px;
                    letter-spacing: 0.5px;
                    pointer-events: none;
                    display: flex;
                    align-items: center;
                    gap: 4px;
                }
                .live-pulse {
                    width: 5px;
                    height: 5px;
                    background: #ef4444;
                    border-radius: 50%;
                    animation: pulse 1.2s infinite;
                }
                @keyframes pulse {
                    0% { opacity: 1; transform: scale(1); }
                    50% { opacity: 0.4; transform: scale(1.3); }
                    100% { opacity: 1; transform: scale(1); }
                }
            </style>
            <div class="timeline-container">
                <div class="timeline-labels">
                    <div class="header-placeholder">Channels</div>
                    <div class="label user"><span class="dot"></span>User</div>
                    <div class="label ai"><span class="dot"></span>MiniCPM-o</div>
                    <div class="label expert"><span class="dot"></span>Expert/Tool</div>
                </div>
                <div class="timeline-scroll">
                    <canvas></canvas>
                    <div class="live-badge"><span class="live-pulse"></span>LIVE</div>
                </div>
            </div>
        `;

        this.canvas = this.shadowRoot.querySelector('canvas');
        this.ctx = this.canvas.getContext('2d');
        
        // State
        this.startTime = 0;
        this.currentTime = 0;
        this.isRunning = false;
        
        // Data tracks
        this.userAudio = []; // { t, rms }
        this.aiAudio = []; // { t, rms }
        this.expertTasks = []; // { id, name, startT, endT, status: 'thinking'|'done'|'error' }
        this.arrows = []; // { from: [lane, t], to: [lane, t] }
        this.bargeIns = []; // { t }

        // Layout config
        this.pixelsPerSecond = 55;
        this.headerHeight = 24; // Top timeline axis
        this.laneHeight = 0; // Calculated dynamically on resize

        this.resizeObserver = new ResizeObserver(() => this.resize());
        this.resizeObserver.observe(this);
    }

    connectedCallback() {
        this.resize();
    }

    disconnectedCallback() {
        this.stop();
        this.resizeObserver.disconnect();
    }

    resize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        if (!rect.width || !rect.height) return;
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = Math.round(rect.width * dpr);
        this.canvas.height = Math.round(rect.height * dpr);
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        
        this.laneHeight = (rect.height - this.headerHeight) / 3;
        if (!this.isRunning) this.draw();
    }

    // --- Public API ---
    start() {
        this.startTime = performance.now();
        this.currentTime = 0;
        this.userAudio = [];
        this.aiAudio = [];
        this.expertTasks = [];
        this.arrows = [];
        this.bargeIns = [];
        this.isRunning = true;
        this.loop();
    }

    stop() {
        this.isRunning = false;
    }

    reset() {
        this.currentTime = 0;
        this.userAudio = [];
        this.aiAudio = [];
        this.expertTasks = [];
        this.arrows = [];
        this.bargeIns = [];
        if (!this.isRunning) this.draw();
    }

    addUserAudio(rms) {
        if (!this.isRunning) return;
        this.userAudio.push({ t: this.currentTime, rms: Math.max(0, Math.min(1, rms)) });
    }

    addAiAudio(rms) {
        if (!this.isRunning) return;
        this.aiAudio.push({ t: this.currentTime, rms: Math.max(0, Math.min(1, rms)) });
    }

    startExpertTask(name) {
        if (!this.isRunning) return null;
        const id = Math.random().toString(36).substr(2, 9);
        this.expertTasks.push({ id, name, startT: this.currentTime, endT: null, status: 'thinking' });
        this.arrows.push({ from: ['ai', this.currentTime], to: ['expert', this.currentTime] });
        return id;
    }

    endExpertTask(id, success = true) {
        if (!this.isRunning) return;
        const task = this.expertTasks.find(t => t.id === id);
        if (task) {
            task.endT = this.currentTime;
            task.status = success ? 'done' : 'error';
            this.arrows.push({ from: ['expert', this.currentTime], to: ['ai', this.currentTime] });
        }
    }

    recordBargeIn() {
        if (!this.isRunning) return;
        this.bargeIns.push({ t: this.currentTime });
    }

    // --- Internal Render Loop ---
    loop() {
        if (!this.isRunning) return;
        const now = performance.now();
        this.currentTime = (now - this.startTime) / 1000;
        this.draw();
        requestAnimationFrame(() => this.loop());
    }

    draw() {
        const width = this.canvas.width / (window.devicePixelRatio || 1);
        const height = this.canvas.height / (window.devicePixelRatio || 1);
        const ctx = this.ctx;
        if (!width || !height) return;

        ctx.clearRect(0, 0, width, height);

        const viewDuration = width / this.pixelsPerSecond;
        let startT = 0;
        const maxScrubberX = width * 0.78;
        if (this.currentTime * this.pixelsPerSecond > maxScrubberX) {
            startT = this.currentTime - (maxScrubberX / this.pixelsPerSecond);
        }

        ctx.save();
        
        // Draw grid, axis & backgrounds
        this.drawBackground(ctx, width, height, startT, viewDuration);
        
        // Lane coordinates
        const laneYs = {
            user: this.headerHeight,
            ai: this.headerHeight + this.laneHeight,
            expert: this.headerHeight + this.laneHeight * 2
        };

        // Draw waveforms
        this.drawWaveform(ctx, this.userAudio, startT, laneYs.user, '#34d399', 'rgba(52, 211, 153, 0.25)');
        this.drawWaveform(ctx, this.aiAudio, startT, laneYs.ai, '#60a5fa', 'rgba(96, 165, 250, 0.25)');
        
        // Draw Expert tasks
        this.drawExpertTasks(ctx, startT, laneYs.expert);

        // Draw Bezier arrows
        this.drawArrows(ctx, startT, laneYs);

        // Draw Barge-in markers
        this.drawBargeIns(ctx, startT, height);

        // Draw Scrubber
        const scrubberX = (this.currentTime - startT) * this.pixelsPerSecond;
        if (scrubberX >= 0 && scrubberX <= width) {
            ctx.beginPath();
            ctx.moveTo(scrubberX, this.headerHeight);
            ctx.lineTo(scrubberX, height);
            ctx.strokeStyle = '#ef4444';
            ctx.lineWidth = 1.5;
            ctx.stroke();
            
            // Scrubber time bubble
            ctx.fillStyle = '#ef4444';
            ctx.beginPath();
            ctx.roundRect(scrubberX - 18, 4, 36, 15, 3);
            ctx.fill();
            
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 9px monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(this.formatTime(this.currentTime), scrubberX, 11.5);
        }

        ctx.restore();
    }

    drawBackground(ctx, w, h, startT, viewDuration) {
        // Alternating row highlights
        ctx.fillStyle = 'rgba(255, 255, 255, 0.015)';
        ctx.fillRect(0, this.headerHeight + this.laneHeight, w, this.laneHeight); // AI lane

        // Grid lines & tick labels
        ctx.strokeStyle = '#1e293b';
        ctx.lineWidth = 1;
        ctx.font = '9px monospace';
        ctx.fillStyle = '#64748b';
        ctx.textAlign = 'left';

        const firstTick = Math.floor(startT);
        const lastTick = Math.ceil(startT + viewDuration);
        
        for (let t = firstTick; t <= lastTick; t++) {
            const x = (t - startT) * this.pixelsPerSecond;
            if (x >= 0 && x <= w) {
                ctx.beginPath();
                ctx.moveTo(x, this.headerHeight - 4);
                ctx.lineTo(x, h);
                ctx.stroke();
                
                if (t % 5 === 0) {
                    ctx.fillText(this.formatTime(t), x + 3, 12);
                }
            }
        }
        
        // Dividers
        ctx.beginPath();
        ctx.moveTo(0, this.headerHeight); ctx.lineTo(w, this.headerHeight);
        ctx.moveTo(0, this.headerHeight + this.laneHeight); ctx.lineTo(w, this.headerHeight + this.laneHeight);
        ctx.moveTo(0, this.headerHeight + this.laneHeight * 2); ctx.lineTo(w, this.headerHeight + this.laneHeight * 2);
        ctx.strokeStyle = '#1e293b';
        ctx.stroke();
    }

    drawWaveform(ctx, data, startT, yOffset, strokeColor, fillColor) {
        if (!data || data.length === 0) return;
        const centerY = yOffset + (this.laneHeight / 2);
        const maxAmp = (this.laneHeight / 2) * 0.85;

        ctx.save();
        ctx.strokeStyle = strokeColor;
        ctx.fillStyle = fillColor;
        ctx.lineWidth = 1.5;

        for (let i = 0; i < data.length; i++) {
            const pt = data[i];
            if (pt.t < startT) continue;
            
            const x = (pt.t - startT) * this.pixelsPerSecond;
            const h = Math.max(1.5, pt.rms * maxAmp);
            
            // Rounded bar
            ctx.beginPath();
            ctx.roundRect(x - 1, centerY - h, 2, h * 2, 1);
            ctx.fill();
        }
        ctx.restore();
    }

    drawExpertTasks(ctx, startT, yOffset) {
        const padding = 8;
        const h = this.laneHeight - (padding * 2);
        
        ctx.font = '11px -apple-system, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        
        for (const task of this.expertTasks) {
            const endT = task.endT || this.currentTime;
            if (endT < startT) continue;
            
            const x = (task.startT - startT) * this.pixelsPerSecond;
            let width = Math.max(16, (endT - task.startT) * this.pixelsPerSecond);
            const y = yOffset + padding;
            
            ctx.save();
            if (task.status === 'thinking') {
                ctx.fillStyle = 'rgba(245, 158, 11, 0.25)';
                ctx.strokeStyle = '#f59e0b';
                ctx.shadowColor = 'rgba(245, 158, 11, 0.6)';
                ctx.shadowBlur = 8;
            } else if (task.status === 'done') {
                ctx.fillStyle = 'rgba(16, 185, 129, 0.2)';
                ctx.strokeStyle = '#10b981';
            } else {
                ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
                ctx.strokeStyle = '#ef4444';
            }

            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.roundRect(x, y, width, h, 6);
            ctx.fill();
            ctx.stroke();
            ctx.restore();

            // Task label
            if (width > 35) {
                ctx.save();
                ctx.beginPath();
                ctx.roundRect(x, y, width, h, 6);
                ctx.clip();
                ctx.fillStyle = '#f8fafc';
                ctx.font = 'bold 10px sans-serif';
                const label = task.status === 'thinking' ? `⚡ ${task.name}...` : `✓ ${task.name}`;
                ctx.fillText(label, x + width / 2, y + h / 2);
                ctx.restore();
            }
        }
    }

    drawArrows(ctx, startT, laneYs) {
        ctx.save();
        ctx.lineWidth = 1.5;
        
        for (const arrow of this.arrows) {
            if (arrow.from[1] < startT && arrow.to[1] < startT) continue;
            
            const startX = (arrow.from[1] - startT) * this.pixelsPerSecond;
            const endX = (arrow.to[1] - startT) * this.pixelsPerSecond;
            
            const getCenterY = (lane) => laneYs[lane] + (this.laneHeight / 2);
            const startY = getCenterY(arrow.from[0]);
            const endY = getCenterY(arrow.to[0]);

            const isDown = startY < endY;
            ctx.strokeStyle = isDown ? '#f59e0b' : '#34d399';
            ctx.fillStyle = ctx.strokeStyle;

            ctx.beginPath();
            ctx.moveTo(startX, startY);
            
            if (Math.abs(startX - endX) < 3) {
                ctx.lineTo(startX, endY);
                this.drawArrowhead(ctx, startX, endY, isDown ? Math.PI/2 : -Math.PI/2);
            } else {
                ctx.bezierCurveTo(startX + 18, startY, endX - 18, endY, endX, endY);
                const angle = Math.atan2(endY - startY, endX - startX);
                this.drawArrowhead(ctx, endX, endY, angle);
            }
            ctx.stroke();
        }
        ctx.restore();
    }

    drawBargeIns(ctx, startT, height) {
        if (!this.bargeIns || this.bargeIns.length === 0) return;
        ctx.save();
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 1.2;

        for (const b of this.bargeIns) {
            if (b.t < startT) continue;
            const x = (b.t - startT) * this.pixelsPerSecond;
            ctx.beginPath();
            ctx.moveTo(x, this.headerHeight);
            ctx.lineTo(x, height);
            ctx.stroke();

            // Interruption flag badge
            ctx.fillStyle = '#ef4444';
            ctx.beginPath();
            ctx.roundRect(x - 3, this.headerHeight + 2, 6, 8, 2);
            ctx.fill();
        }
        ctx.restore();
    }

    drawArrowhead(ctx, x, y, angle) {
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(angle);
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(-5, -3.5);
        ctx.lineTo(-5, 3.5);
        ctx.fill();
        ctx.restore();
    }

    formatTime(seconds) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    }
}

customElements.define('gpt-timeline', GPTTimeline);
