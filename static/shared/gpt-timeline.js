export class GPTTimeline extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.shadowRoot.innerHTML = `
            <style>
                :host {
                    display: block;
                    width: 100%;
                    height: 200px;
                    background: #111;
                    color: #fff;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    position: relative;
                    border-radius: 8px;
                    overflow: hidden;
                    box-sizing: border-box;
                    border: 1px solid #333;
                }
                .timeline-container {
                    display: flex;
                    width: 100%;
                    height: 100%;
                }
                .timeline-labels {
                    width: 100px;
                    height: 100%;
                    background: #1a1a1a;
                    display: flex;
                    flex-direction: column;
                    border-right: 1px solid #333;
                    z-index: 10;
                    flex-shrink: 0;
                }
                .label {
                    flex: 1;
                    display: flex;
                    align-items: center;
                    padding-left: 10px;
                    font-size: 12px;
                    font-weight: 600;
                    letter-spacing: 0.5px;
                    border-bottom: 1px solid #2a2a2a;
                }
                .label:last-child {
                    border-bottom: none;
                }
                .label.user { color: #4ade80; }
                .label.ai { color: #60a5fa; }
                .label.expert { color: #fb923c; }
                
                .timeline-scroll {
                    flex: 1;
                    height: 100%;
                    position: relative;
                    overflow: hidden; /* canvas will handle scrolling internally */
                }
                canvas {
                    display: block;
                    width: 100%;
                    height: 100%;
                }
                .time-indicator {
                    position: absolute;
                    top: 0;
                    left: 0;
                    background: rgba(255, 255, 255, 0.8);
                    color: #000;
                    font-size: 10px;
                    padding: 2px 4px;
                    border-radius: 4px;
                    pointer-events: none;
                    transform: translateX(-50%);
                    z-index: 20;
                    display: none;
                }
            </style>
            <div class="timeline-container">
                <div class="timeline-labels">
                    <div class="label user">User</div>
                    <div class="label ai">MiniCPM-o</div>
                    <div class="label expert">Expert</div>
                </div>
                <div class="timeline-scroll">
                    <canvas></canvas>
                    <div class="time-indicator" id="timeIndicator">0:00</div>
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
        this.expertTasks = []; // { id, name, startT, endT, status: 'thinking'|'done' }
        this.arrows = []; // { from: [lane, t], to: [lane, t] }

        // Layout config
        this.pixelsPerSecond = 50;
        this.laneHeight = 0; // Calculated on resize
        this.headerHeight = 20; // Top axis

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
        this.canvas.width = rect.width * window.devicePixelRatio;
        this.canvas.height = rect.height * window.devicePixelRatio;
        this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        
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
        this.isRunning = true;
        this.loop();
    }

    stop() {
        this.isRunning = false;
    }

    // Call this frequently with RMS value (0.0 to 1.0)
    addUserAudio(rms) {
        if (!this.isRunning) return;
        this.userAudio.push({ t: this.currentTime, rms });
    }

    addAiAudio(rms) {
        if (!this.isRunning) return;
        this.aiAudio.push({ t: this.currentTime, rms });
    }

    startExpertTask(name) {
        if (!this.isRunning) return null;
        const id = Math.random().toString(36).substr(2, 9);
        this.expertTasks.push({ id, name, startT: this.currentTime, endT: null, status: 'thinking' });
        // Arrow from AI to Expert
        this.arrows.push({ from: ['ai', this.currentTime], to: ['expert', this.currentTime] });
        return id;
    }

    endExpertTask(id, success = true) {
        if (!this.isRunning) return;
        const task = this.expertTasks.find(t => t.id === id);
        if (task) {
            task.endT = this.currentTime;
            task.status = success ? 'done' : 'error';
            // Arrow from Expert back to AI
            this.arrows.push({ from: ['expert', this.currentTime], to: ['ai', this.currentTime] });
        }
    }
    
    // Allows injecting exact time instead of using performance.now
    tick(deltaTimeMs) {
        if (this.isRunning) {
             this.currentTime += deltaTimeMs / 1000;
        }
    }

    // --- Internal ---
    loop() {
        if (!this.isRunning) return;
        
        // Auto-update time if not ticked manually
        const now = performance.now();
        this.currentTime = (now - this.startTime) / 1000;
        
        this.draw();
        requestAnimationFrame(() => this.loop());
    }

    draw() {
        const width = this.canvas.width / window.devicePixelRatio;
        const height = this.canvas.height / window.devicePixelRatio;
        const ctx = this.ctx;

        ctx.clearRect(0, 0, width, height);

        // Determine view window (scroll to keep current time on the right if needed)
        const viewDuration = width / this.pixelsPerSecond;
        let startT = 0;
        
        // Keep scrubber at 80% of width if it exceeds
        const maxScrubberX = width * 0.8;
        if (this.currentTime * this.pixelsPerSecond > maxScrubberX) {
            startT = this.currentTime - (maxScrubberX / this.pixelsPerSecond);
        }

        ctx.save();
        
        // Draw axis & backgrounds
        this.drawBackground(ctx, width, height, startT, viewDuration);
        
        // Draw Lanes
        const laneYs = {
            user: this.headerHeight,
            ai: this.headerHeight + this.laneHeight,
            expert: this.headerHeight + this.laneHeight * 2
        };

        this.drawWaveform(ctx, this.userAudio, startT, laneYs.user, '#4ade80');
        this.drawWaveform(ctx, this.aiAudio, startT, laneYs.ai, '#60a5fa');
        this.drawExpertTasks(ctx, startT, laneYs.expert, '#fb923c');
        this.drawArrows(ctx, startT, laneYs);

        // Draw Scrubber
        const scrubberX = (this.currentTime - startT) * this.pixelsPerSecond;
        if (scrubberX >= 0 && scrubberX <= width) {
            ctx.beginPath();
            ctx.moveTo(scrubberX, this.headerHeight);
            ctx.lineTo(scrubberX, height);
            ctx.strokeStyle = '#ef4444';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            // Draw time bubble on canvas
            ctx.fillStyle = '#ef4444';
            ctx.beginPath();
            ctx.roundRect(scrubberX - 20, 2, 40, 16, 4);
            ctx.fill();
            
            ctx.fillStyle = '#fff';
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(this.formatTime(this.currentTime), scrubberX, 10);
        }

        ctx.restore();
    }

    drawBackground(ctx, w, h, startT, viewDuration) {
        // Draw alternating rows
        ctx.fillStyle = 'rgba(255, 255, 255, 0.02)';
        ctx.fillRect(0, this.headerHeight + this.laneHeight, w, this.laneHeight); // AI lane bg

        // Draw grid lines
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.font = '10px sans-serif';
        ctx.fillStyle = '#666';
        ctx.textAlign = 'left';

        // Tick marks every 1 second
        const firstTick = Math.floor(startT);
        const lastTick = Math.ceil(startT + viewDuration);
        
        for (let t = firstTick; t <= lastTick; t++) {
            const x = (t - startT) * this.pixelsPerSecond;
            if (x >= 0 && x <= w) {
                ctx.beginPath();
                ctx.moveTo(x, this.headerHeight - 5);
                ctx.lineTo(x, h);
                ctx.stroke();
                
                if (t % 5 === 0) { // Label every 5 seconds
                    ctx.fillText(this.formatTime(t), x + 2, 10);
                }
            }
        }
        
        // Draw Lane dividers
        ctx.beginPath();
        ctx.moveTo(0, this.headerHeight); ctx.lineTo(w, this.headerHeight);
        ctx.moveTo(0, this.headerHeight + this.laneHeight); ctx.lineTo(w, this.headerHeight + this.laneHeight);
        ctx.moveTo(0, this.headerHeight + this.laneHeight * 2); ctx.lineTo(w, this.headerHeight + this.laneHeight * 2);
        ctx.strokeStyle = '#222';
        ctx.stroke();
    }

    drawWaveform(ctx, data, startT, yOffset, color) {
        if (data.length === 0) return;
        
        const centerY = yOffset + (this.laneHeight / 2);
        const maxAmp = (this.laneHeight / 2) * 0.8;
        
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        // Simple bar drawing for waveform
        for (let i = 0; i < data.length; i++) {
            const pt = data[i];
            if (pt.t < startT) continue; // Offscreen left
            
            const x = (pt.t - startT) * this.pixelsPerSecond;
            const h = pt.rms * maxAmp;
            
            ctx.moveTo(x, centerY - h);
            ctx.lineTo(x, centerY + h);
        }
        ctx.stroke();
    }

    drawExpertTasks(ctx, startT, yOffset, color) {
        const padding = 10;
        const h = this.laneHeight - (padding * 2);
        
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        
        for (const task of this.expertTasks) {
            const endT = task.endT || this.currentTime; // if not done, draw to current time
            if (endT < startT) continue;
            
            const x = (task.startT - startT) * this.pixelsPerSecond;
            let width = (endT - task.startT) * this.pixelsPerSecond;
            if (width < 2) width = 2; // min width
            
            const y = yOffset + padding;
            
            // Draw block
            ctx.fillStyle = task.status === 'thinking' ? 'rgba(251, 146, 60, 0.5)' : 'rgba(251, 146, 60, 0.8)';
            ctx.beginPath();
            ctx.roundRect(x, y, width, h, 4);
            ctx.fill();
            ctx.strokeStyle = color;
            ctx.lineWidth = 1;
            ctx.stroke();

            // Draw text if enough width
            if (width > 30) {
                ctx.fillStyle = '#fff';
                // Clip text
                ctx.save();
                ctx.beginPath();
                ctx.roundRect(x, y, width, h, 4);
                ctx.clip();
                ctx.fillText(task.name, x + width/2, y + h/2);
                ctx.restore();
            }
        }
    }

    drawArrows(ctx, startT, laneYs) {
        ctx.strokeStyle = '#fff';
        ctx.fillStyle = '#fff';
        ctx.lineWidth = 2;
        
        for (const arrow of this.arrows) {
            if (arrow.from[1] < startT && arrow.to[1] < startT) continue;
            
            const startX = (arrow.from[1] - startT) * this.pixelsPerSecond;
            const endX = (arrow.to[1] - startT) * this.pixelsPerSecond;
            
            // Get center Y of lanes
            const getCenterY = (lane) => laneYs[lane] + (this.laneHeight / 2);
            
            const startY = getCenterY(arrow.from[0]);
            const endY = getCenterY(arrow.to[0]);

            // Draw arrow line
            ctx.beginPath();
            ctx.moveTo(startX, startY);
            
            if (Math.abs(startX - endX) < 2) {
                // Vertical line
                ctx.lineTo(startX, endY);
                this.drawArrowhead(ctx, startX, endY, startY < endY ? Math.PI/2 : -Math.PI/2);
            } else {
                // Curved line (e.g. from Expert back to AI delayed)
                ctx.bezierCurveTo(startX + 20, startY, endX - 20, endY, endX, endY);
                // Angle at end
                const angle = Math.atan2(endY - startY, endX - startX);
                this.drawArrowhead(ctx, endX, endY, angle);
            }
            ctx.stroke();
        }
    }
    
    drawArrowhead(ctx, x, y, angle) {
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(angle);
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(-6, -4);
        ctx.lineTo(-6, 4);
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
