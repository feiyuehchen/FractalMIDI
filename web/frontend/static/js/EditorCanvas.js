/**
 * FractalMIDI Editor Canvas
 * High-performance renderer with sparse data structures and audio integration.
 */

class EditorCanvas {
    constructor(canvasId, containerId) {
        this.canvas = document.getElementById(canvasId);
        this.container = document.getElementById(containerId);
        this.ctx = this.canvas.getContext('2d', { alpha: false });

        // --- Data Model ---
        this.notes = []; // List of {pitch, start, duration, velocity}
        this.noteMap = new Map(); // tick -> [note objects] for fast playback lookup
        this.maskMap = new Map(); // Key: "t,p", Value: true
        
        // --- Viewport State ---
        this.zoomX = 1.0; 
        this.zoomY = 1.0; 
        this.offsetX = 0; 
        this.offsetY = 0; 
        
        // --- Config ---
        this.CONFIG = {
            PITCH_COUNT: 128,
            TICKS_PER_BEAT: 4, // 16th notes
            BASE_NOTE_HEIGHT: 12,
            BASE_NOTE_WIDTH: 20,
            MIN_ZOOM: 0.1,
            MAX_ZOOM: 4.0,
            GRID_COLOR: '#2a2a30',
            BAR_LINE_COLOR: '#555560',
            OCTAVE_LINE_COLOR: '#555560',
            BG_COLOR: '#1a1a20',
            PLAYHEAD_COLOR: '#ffffff',
            MASK_COLOR: 'rgba(255, 71, 87, 0.3)',
            MASK_HATCH_COLOR: 'rgba(255, 71, 87, 0.5)',
            CONDITION_OVERLAY_COLOR: 'rgba(0, 0, 0, 0.6)',
            CONDITION_BORDER_COLOR: '#00f3ff',
            RULER_WIDTH: 50,
            TOP_RULER_HEIGHT: 24,
            RULER_BG: '#121214',
            RULER_TEXT: '#888',
            KEY_WHITE: '#25252b',
            KEY_BLACK: '#151518',
            KEY_BORDER: '#333'
        };

        // --- Interaction State ---
        this.mode = 'view'; 
        this.tool = 'pen'; 
        this.isDragging = false;
        this.isScrubbing = false;
        this.draggingConditionTarget = null;
        this.lastMouseX = 0;
        this.lastMouseY = 0;
        this.brushSize = 1; 
        this.brushColor = 0.8; 
        this.isPlaying = false;
        
        this.eraserMode = 'delete'; // 'delete' or 'mask'
        
        // --- History (Undo/Redo) ---
        this.history = [];
        this.historyIndex = -1;
        this.maxHistory = 50;

        this.conditionRegion = null; // {start: 0, end: 32}
        
        // For continuous drawing
        this.activeNote = null; 
        this.lastPaintedTick = -1;
        this.lastPaintedPitch = -1;
        this.lastPreviewTime = 0;

        // --- Color Map ---
        this.colormap = this.generateColormap();

        // --- Audio ---
        this.synth = null;
        this.initAudio();
        
        // --- Playback Scheduler ---
        this.playbackRepeaterId = null;

        // --- Initialization ---
        this.resize();
        new ResizeObserver(() => this.resize()).observe(this.container);
        window.addEventListener('resize', () => this.resize());
        this.setupEvents();
        
        // Init History
        this.pushState();
        
        this.renderLoop = this.renderLoop.bind(this);
        requestAnimationFrame(this.renderLoop);
    }

    generateColormap() {
        const colors = [];
        const stops = [
            { pos: 0.0, r: 0, g: 0, b: 128 },
            { pos: 0.2, r: 0, g: 128, b: 255 },
            { pos: 0.4, r: 0, g: 255, b: 128 },
            { pos: 0.6, r: 255, g: 255, b: 0 },
            { pos: 0.8, r: 255, g: 128, b: 0 },
            { pos: 1.0, r: 255, g: 0, b: 0 }
        ];

        for (let i = 0; i < 128; i++) {
            const t = i / 127;
            let c1 = stops[0], c2 = stops[stops.length-1];
            for (let j = 0; j < stops.length - 1; j++) {
                if (t >= stops[j].pos && t <= stops[j+1].pos) {
                    c1 = stops[j]; c2 = stops[j+1]; break;
                }
            }
            const localT = (t - c1.pos) / (c2.pos - c1.pos);
            const r = Math.round(c1.r + (c2.r - c1.r) * localT);
            const g = Math.round(c1.g + (c2.g - c1.g) * localT);
            const b = Math.round(c1.b + (c2.b - c1.b) * localT);
            colors.push(`rgb(${r},${g},${b})`);
        }
        return colors;
    }

    async initAudio() {
        if (window.Tone) {
            // Use a Sampler for realistic Piano sound
            // Fallback to synth if samples fail to load (handled by Tone mostly)
            this.synth = new Tone.Sampler({
                urls: {
                    "A0": "A0.mp3",
                    "C1": "C1.mp3",
                    "D#1": "Ds1.mp3",
                    "F#1": "Fs1.mp3",
                    "A1": "A1.mp3",
                    "C2": "C2.mp3",
                    "D#2": "Ds2.mp3",
                    "F#2": "Fs2.mp3",
                    "A2": "A2.mp3",
                    "C3": "C3.mp3",
                    "D#3": "Ds3.mp3",
                    "F#3": "Fs3.mp3",
                    "A3": "A3.mp3",
                    "C4": "C4.mp3",
                    "D#4": "Ds4.mp3",
                    "F#4": "Fs4.mp3",
                    "A4": "A4.mp3",
                    "C5": "C5.mp3",
                    "D#5": "Ds5.mp3",
                    "F#5": "Fs5.mp3",
                    "A5": "A5.mp3",
                    "C6": "C6.mp3",
                    "D#6": "Ds6.mp3",
                    "F#6": "Fs6.mp3",
                    "A6": "A6.mp3",
                    "C7": "C7.mp3",
                    "D#7": "Ds7.mp3",
                    "F#7": "Fs7.mp3",
                    "A7": "A7.mp3",
                    "C8": "C8.mp3"
                },
                baseUrl: "https://tonejs.github.io/audio/salamander/",
                release: 1,
                onload: () => console.log("Piano Samples Loaded")
            }).toDestination();
            
            this.synth.volume.value = -5;
            console.log("Audio Engine Initialized (Piano Sampler)");
        }
    }
    
    setBPM(bpm) {
        if (window.Tone) Tone.Transport.bpm.value = bpm;
    }

    resize() {
        const rect = this.container.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.canvas.style.width = `${rect.width}px`;
        this.canvas.style.height = `${rect.height}px`;
        
        this.ctx.setTransform(1, 0, 0, 1, 0, 0); // Reset transform to prevent accumulation
        this.ctx.scale(dpr, dpr);
        
        if (this.zoomY === 1.0 && this.offsetY === 0) {
            // Default view logic
            this.fitView();
        }
        this.render();
    }
    
    fitView() {
        const rect = this.container.getBoundingClientRect();
        const w = rect.width - this.CONFIG.RULER_WIDTH;
        const h = rect.height - this.CONFIG.TOP_RULER_HEIGHT;
        
        // Determine content bounds
        let maxTick = 256;
        if (this.notes.length > 0) {
            maxTick = Math.max(...this.notes.map(n => n.start + n.duration));
            maxTick = Math.max(maxTick, 64); // Min width
        }
        
        // Calc zoomX
        // Fit maxTick into w
        this.zoomX = w / (maxTick * this.CONFIG.BASE_NOTE_WIDTH);
        if (this.zoomX > this.CONFIG.MAX_ZOOM) this.zoomX = this.CONFIG.MAX_ZOOM;
        if (this.zoomX < this.CONFIG.MIN_ZOOM) this.zoomX = this.CONFIG.MIN_ZOOM;
        
        // Calc zoomY to fit 128 pitches
        this.zoomY = h / (this.CONFIG.PITCH_COUNT * this.CONFIG.BASE_NOTE_HEIGHT);
        
        // Recalc cell size
        this.cellW = this.CONFIG.BASE_NOTE_WIDTH * this.zoomX;
        this.cellH = this.CONFIG.BASE_NOTE_HEIGHT * this.zoomY;
        
        // Reset offsets
        this.offsetX = 0;
        this.offsetY = this.CONFIG.TOP_RULER_HEIGHT; // Start below top ruler
        
        this.render();
    }

    pitchToY(pitch) { return (127 - pitch) * this.cellH + this.offsetY; }
    yToPitch(y) { return 127 - Math.floor((y - this.offsetY) / this.cellH); }
    timeToX(tick) { return tick * this.cellW + this.offsetX + this.CONFIG.RULER_WIDTH; }
    xToTime(x) { return Math.floor((x - this.offsetX - this.CONFIG.RULER_WIDTH) / this.cellW); }

    renderLoop() {
        if (this.isPlaying || (window.Tone && Tone.Transport.state === 'started')) {
            this.render();
        }
        requestAnimationFrame(this.renderLoop);
    }

    render() {
        const rect = this.canvas.getBoundingClientRect();
        const w = rect.width; // logical width
        const h = rect.height; // logical height
        const th = this.CONFIG.TOP_RULER_HEIGHT;
        
        const ctx = this.ctx;
        
        this.cellW = this.CONFIG.BASE_NOTE_WIDTH * this.zoomX;
        this.cellH = this.CONFIG.BASE_NOTE_HEIGHT * this.zoomY;

        // Clear
        ctx.fillStyle = this.CONFIG.BG_COLOR;
        ctx.fillRect(0, 0, w, h);

        const startTick = Math.floor(-(this.offsetX) / this.cellW);
        const endTick = startTick + Math.ceil((w - this.CONFIG.RULER_WIDTH) / this.cellW) + 1;
        const startPitch = Math.floor(this.yToPitch(h));
        const endPitch = Math.ceil(this.yToPitch(th));

        // Grid
        ctx.lineWidth = 1; // High DPI adjustment? usually 1 is fine
        
        for (let p = Math.max(0, startPitch); p <= Math.min(127, endPitch); p++) {
            const y = Math.floor(this.pitchToY(p)) + 0.5; // snap to pixel
            ctx.strokeStyle = (p % 12 === 0) ? this.CONFIG.OCTAVE_LINE_COLOR : this.CONFIG.GRID_COLOR;
            ctx.beginPath(); ctx.moveTo(this.CONFIG.RULER_WIDTH, y); ctx.lineTo(w, y); ctx.stroke();
        }

        for (let t = Math.max(0, startTick); t <= endTick; t++) {
            const x = Math.floor(this.timeToX(t)) + 0.5;
            if (x < this.CONFIG.RULER_WIDTH) continue;
            ctx.strokeStyle = (t % 16 === 0) ? this.CONFIG.BAR_LINE_COLOR : this.CONFIG.GRID_COLOR;
            ctx.beginPath(); ctx.moveTo(x, th); ctx.lineTo(x, h); ctx.stroke();
        }

        // Notes
        for (const note of this.notes) {
            if (note.start > endTick || (note.start + note.duration) < startTick) continue;
            if (note.pitch > endPitch + 2 || note.pitch < startPitch - 2) continue;

            const x = Math.floor(this.timeToX(note.start));
            const y = Math.floor(this.pitchToY(note.pitch));
            const nw = Math.max(1, Math.ceil(note.duration * this.cellW)) - 1; 
            const nh = Math.ceil(this.cellH) - 1;
            
            if (x + nw < this.CONFIG.RULER_WIDTH) continue;
            if (y + nh < th) continue;

            const velIdx = Math.floor(note.velocity * 127);
            ctx.fillStyle = this.colormap[Math.min(127, Math.max(0, velIdx))];
            ctx.fillRect(x, y, nw, nh);
            
            ctx.fillStyle = 'rgba(255,255,255,0.3)';
            ctx.fillRect(x, y, nw, 2);
        }

        // Mask
        if (this.maskMap.size > 0) {
            ctx.fillStyle = this.CONFIG.MASK_COLOR;
            for (let t = Math.max(0, startTick); t <= endTick; t++) {
                for (let p = Math.max(0, startPitch); p <= Math.min(127, endPitch); p++) {
                    if (this.maskMap.has(`${t},${p}`)) {
                        const x = Math.floor(this.timeToX(t));
                        const y = Math.floor(this.pitchToY(p));
                        if (x < this.CONFIG.RULER_WIDTH) continue;
                        if (y < th) continue;
                        
                        ctx.fillRect(x, y, Math.ceil(this.cellW), Math.ceil(this.cellH));
                        
                        ctx.strokeStyle = this.CONFIG.MASK_HATCH_COLOR;
                        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + this.cellW, y + this.cellH); ctx.stroke();
                    }
                }
            }
        }
        
        // Condition Region Overlay
        if (this.conditionRegion) {
            const { start, end } = this.conditionRegion;
            
            // Draw dark overlay outside of condition region
            // Left side (before start)
            if (start > startTick) {
                const x1 = Math.max(this.CONFIG.RULER_WIDTH, this.timeToX(startTick));
                const x2 = this.timeToX(start);
                if (x2 > x1) {
                    ctx.fillStyle = this.CONFIG.CONDITION_OVERLAY_COLOR;
                    ctx.fillRect(x1, th, x2 - x1, h - th);
                }
            }
            
            // Right side (after end)
            if (end < endTick) {
                const x1 = this.timeToX(end);
                const x2 = Math.min(w, this.timeToX(endTick));
                if (x2 > x1) {
                    ctx.fillStyle = this.CONFIG.CONDITION_OVERLAY_COLOR;
                    ctx.fillRect(x1, th, x2 - x1, h - th);
                }
            }
            
            // Highlight borders & Draggable Handle
            ctx.strokeStyle = this.CONFIG.CONDITION_BORDER_COLOR;
            ctx.lineWidth = 2;
            
            // Start Line
            const xStart = this.timeToX(start);
            if (xStart >= this.CONFIG.RULER_WIDTH) {
                ctx.beginPath();
                ctx.moveTo(xStart, th); ctx.lineTo(xStart, h);
                ctx.stroke();
            }

            // End Line
            const xEnd = this.timeToX(end);
            if (xEnd >= this.CONFIG.RULER_WIDTH) {
                ctx.beginPath();
                ctx.moveTo(xEnd, th); ctx.lineTo(xEnd, h);
                ctx.stroke();
            }
            
            // Label
            ctx.font = 'bold 12px Rajdhani'; 
            ctx.fillStyle = this.CONFIG.CONDITION_BORDER_COLOR;
            ctx.fillText("CONDITION", Math.max(xStart, this.CONFIG.RULER_WIDTH) + 5, th + 15);
        }
        
        // Draw Rulers
        this.drawSideRuler(startPitch, endPitch, w, h);
        this.drawTopRuler(startTick, endTick, w, h);

        // Playhead
        if (window.Tone) {
             const ticks = Tone.Transport.ticks; 
             const current16th = ticks / (Tone.Transport.PPQ / 4);
             const x = this.timeToX(current16th);
             
             if (x >= this.CONFIG.RULER_WIDTH && x <= w) {
                 ctx.strokeStyle = this.CONFIG.PLAYHEAD_COLOR;
                 ctx.lineWidth = 2;
                 ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
                 
                 ctx.fillStyle = this.CONFIG.PLAYHEAD_COLOR;
                 ctx.beginPath();
                 ctx.moveTo(x - 6, 0);
                 ctx.lineTo(x + 6, 0);
                 ctx.lineTo(x, 12);
                 ctx.fill();
             }
             
             if (this.isPlaying && x > w * 0.9) {
                 this.offsetX -= w * 0.5;
             }
        }
    }
    
    drawSideRuler(startPitch, endPitch, w, h) {
        const ctx = this.ctx;
        const rw = this.CONFIG.RULER_WIDTH;
        const th = this.CONFIG.TOP_RULER_HEIGHT;
        
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, th, rw, h - th);
        ctx.clip();
        
        ctx.fillStyle = this.CONFIG.RULER_BG;
        ctx.fillRect(0, th, rw, h - th);
        
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.font = '10px Rajdhani'; // Modern font
        
        const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
        
        for (let p = Math.max(0, startPitch); p <= Math.min(127, endPitch); p++) {
            const y = Math.floor(this.pitchToY(p));
            const name = NOTE_NAMES[p % 12];
            const octave = Math.floor(p / 12) - 1;
            const isBlack = name.includes('#');
            
            // Draw Key (Flat design)
            ctx.fillStyle = isBlack ? this.CONFIG.KEY_BLACK : this.CONFIG.KEY_WHITE;
            ctx.fillRect(0, y, rw, Math.ceil(this.cellH));
            
            // Border
            ctx.fillStyle = this.CONFIG.KEY_BORDER;
            ctx.fillRect(0, y + Math.ceil(this.cellH) - 1, rw, 1);
            
            // Label C notes
            if (name === 'C') {
                ctx.fillStyle = '#aaa';
                ctx.fillText(`${name}${octave}`, rw - 4, y + this.cellH/2);
            }
        }
        
        // Right border of ruler
        ctx.strokeStyle = '#333';
        ctx.beginPath(); ctx.moveTo(rw, th); ctx.lineTo(rw, h); ctx.stroke();
        
        ctx.restore();
    }

    drawTopRuler(startTick, endTick, w, h) {
        const ctx = this.ctx;
        const th = this.CONFIG.TOP_RULER_HEIGHT;
        const rw = this.CONFIG.RULER_WIDTH;

        ctx.fillStyle = this.CONFIG.RULER_BG;
        ctx.fillRect(0, 0, w, th);
        ctx.strokeStyle = '#333';
        ctx.beginPath(); ctx.moveTo(0, th); ctx.lineTo(w, th); ctx.stroke();
        
        ctx.fillStyle = '#222';
        ctx.fillRect(0, 0, rw, th);
        
        ctx.fillStyle = this.CONFIG.RULER_TEXT;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'bottom';
        ctx.font = '10px Rajdhani';
        
        for (let t = Math.max(0, startTick); t <= endTick; t++) {
            const x = Math.floor(this.timeToX(t)) + 0.5;
            if (x < rw) continue;
            
            if (t % 16 === 0) {
                const bar = t / 16 + 1;
                ctx.strokeStyle = '#666';
                ctx.beginPath(); ctx.moveTo(x, th/2); ctx.lineTo(x, th); ctx.stroke();
                ctx.fillText(bar.toString(), x + 2, th - 2);
            } else if (t % 4 === 0) {
                ctx.strokeStyle = '#444';
                ctx.beginPath(); ctx.moveTo(x, th*0.7); ctx.lineTo(x, th); ctx.stroke();
            }
        }
    }

    // --- Interaction ---

    setupEvents() {
        const c = this.canvas;
        c.addEventListener('mousedown', e => this.handleMouseDown(e));
        window.addEventListener('mousemove', e => this.handleMouseMove(e));
        window.addEventListener('mouseup', e => this.handleMouseUp(e));
        c.addEventListener('wheel', e => this.handleWheel(e));
        c.addEventListener('contextmenu', e => e.preventDefault());
    }

    handleMouseDown(e) {
        if (e.button !== 0) return;
        this.isDragging = true;
        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;

        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // 0. Check Condition Region Drag
        if (this.conditionRegion) {
            const xStart = this.timeToX(this.conditionRegion.start);
            const xEnd = this.timeToX(this.conditionRegion.end);
            
            if (Math.abs(x - xStart) < 10) {
                this.draggingConditionTarget = 'start';
                return;
            } else if (Math.abs(x - xEnd) < 10) {
                this.draggingConditionTarget = 'end';
                return;
            }
        }
        
        // 1. Top Ruler Click (Scrubbing)
        if (y < this.CONFIG.TOP_RULER_HEIGHT && x > this.CONFIG.RULER_WIDTH) {
            const t = this.xToTime(x);
            if (t >= 0 && window.Tone) {
                const ticks = t * (Tone.Transport.PPQ / 4);
                Tone.Transport.ticks = ticks;
                this.render();
            }
            this.isScrubbing = true;
            return;
        }

        // 2. Side Ruler Click (Preview)
        if (x < this.CONFIG.RULER_WIDTH && y > this.CONFIG.TOP_RULER_HEIGHT) {
            const p = this.yToPitch(y);
            if (p >= 0 && p <= 127) this.synth?.triggerAttackRelease(Tone.Frequency(p, "midi"), "8n");
            return;
        }
        
        // 3. Canvas Click (Painting)
        const t = this.xToTime(x);
        const p = this.yToPitch(y);
        
        if (this.mode === 'edit') {
            this.activeNote = null;
            this.lastPaintedTick = -1;
            this.lastPaintedPitch = -1;
            this.applyTool(t, p, true);
        }
    }

    handleMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Cursor logic for Condition
        let hoverCond = false;
        if (this.conditionRegion) {
             if (Math.abs(x - this.timeToX(this.conditionRegion.start)) < 10 || 
                 Math.abs(x - this.timeToX(this.conditionRegion.end)) < 10) {
                 hoverCond = true;
             }
        }
        
        if (hoverCond) {
            this.canvas.style.cursor = 'ew-resize';
        } else if (y < this.CONFIG.TOP_RULER_HEIGHT && x > this.CONFIG.RULER_WIDTH) {
            this.canvas.style.cursor = 'pointer';
        } else {
            this.canvas.style.cursor = 'default';
        }

        if (!this.isDragging) return;
        
        const dx = e.clientX - this.lastMouseX;
        const dy = e.clientY - this.lastMouseY;

        // Condition Dragging
        if (this.draggingConditionTarget) {
            let t = Math.max(0, this.xToTime(x)); // Clamp to positive time
            
            if (this.draggingConditionTarget === 'start') {
                // Ensure start < end
                if (t < this.conditionRegion.end) {
                    this.conditionRegion.start = t;
                }
            } else {
                // Ensure end > start
                if (t > this.conditionRegion.start) {
                    this.conditionRegion.end = t;
                    // Update length slider
                    const slider = document.getElementById('cond-length');
                    if(slider) {
                        slider.value = t - this.conditionRegion.start;
                        slider.dispatchEvent(new Event('input'));
                    }
                }
            }
            this.render();
            this.lastMouseX = e.clientX;
            return;
        }

        if (this.isScrubbing) {
             const t = this.xToTime(x);
             if (t >= 0 && window.Tone) {
                const ticks = t * (Tone.Transport.PPQ / 4);
                Tone.Transport.ticks = ticks;
                this.render();
             }
             this.lastMouseX = e.clientX;
             this.lastMouseY = e.clientY;
             return;
        }

        if (this.mode === 'view') {
            this.offsetX += dx;
            // Clamp offsetX to prevent dragging Bar 0 rightwards (away from left edge)
            if (this.offsetX > 0) this.offsetX = 0;
            
            this.offsetY += dy;
            this.render();
        } else if (this.mode === 'edit') {
            if (x < this.CONFIG.RULER_WIDTH || y < this.CONFIG.TOP_RULER_HEIGHT) return;
            const t = this.xToTime(x);
            const p = this.yToPitch(y);
            this.applyTool(t, p, false);
        }
        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;
    }

    handleMouseUp(e) {
        this.isDragging = false;
        this.isScrubbing = false;
        this.draggingConditionTarget = null;
        this.activeNote = null;
    }

    handleWheel(e) {
        e.preventDefault();
        const zoomSpeed = 0.1;
        const delta = e.deltaY > 0 ? (1 - zoomSpeed) : (1 + zoomSpeed);
        if (e.shiftKey) this.zoomY *= delta;
        else this.zoomX *= delta;
        this.render();
    }

    applyTool(t, p, isStart) {
        if (t < 0 || p < 0 || p > 127) return;
        
        // Save state on start of action (PUSH OLD STATE)
        if (isStart) {
            this.pushState();
        }
        
        if (this.tool === 'pen') {
            if (this.brushSize === 1) {
                this.paintSingleNoteSmart(t, p, isStart);
            } else {
                 for (let dt = 0; dt < this.brushSize; dt++) {
                    for (let dp = 0; dp < this.brushSize; dp++) {
                        this.paintNoteSimple(t + dt, p - dp);
                    }
                }
            }
        } else if (this.tool === 'eraser') {
            for (let dt = 0; dt < this.brushSize; dt++) {
                for (let dp = 0; dp < this.brushSize; dp++) {
                    this.handleEraser(t + dt, p - dp);
                }
            }
            this.render();
        }
        
        // If it's the end of a drag (mouseup), we should probably ensure state is clean?
        // Currently we push at start, so current changes modify "current state" which is technically not saved until next push?
        // Actually, standard undo:
        // 1. Start state S0. Index=0.
        // 2. Modifying... S0 is modified? NO.
        // We need to push S0 to history, then modify active state.
        // So if we are at index 0. We push S0. History=[S0, S0]. Index=1. We modify S1.
        // Correct.
    }

    handleEraser(t, p) {
        if (this.eraserMode === 'mask') {
            // Mark as mask (Regenerate)
            // We typically keep the note underneath as a "suggestion" or just noise, 
            // but usually inpainting ignores the content of masked area unless configured otherwise.
            // For visual clarity, let's keep the note but overlay the mask.
            this.maskMap.set(`${t},${p}`, true);
        } else {
            // Delete mode: Remove note and Unmask (explicitly empty)
            this.maskMap.delete(`${t},${p}`);
            const idx = this.notes.findIndex(n => n.start <= t && n.start + n.duration > t && n.pitch === p);
            if (idx !== -1) {
                 this.notes.splice(idx, 1);
                 this.updateNoteMap();
            }
        }
    }

    // Replaces paintSingleNoteSmart, paintNoteSimple, eraseMask with updated logic if needed,
    // but existing paint logic is fine. eraseMask is replaced by handleEraser.

    paintSingleNoteSmart(t, p, isStart) {
        if (!isStart && this.activeNote && this.activeNote.pitch === p) {
            if (t > this.lastPaintedTick) {
                const diff = t - this.lastPaintedTick;
                this.activeNote.duration += diff;
                this.lastPaintedTick = t;
                this.updateNoteMap();
                this.render();
            }
            return; 
        }

        this.maskMap.delete(`${t},${p}`);
        
        const existingIdx = this.notes.findIndex(n => n.start <= t && n.start + n.duration > t && n.pitch === p);
        let changed = false;
        if (existingIdx !== -1) {
             this.notes.splice(existingIdx, 1);
             changed = true;
        }

        if (this.brushColor > 0.01) {
            const newNote = { pitch: p, start: t, duration: 1, velocity: this.brushColor };
            this.notes.push(newNote);
            this.activeNote = newNote;
            this.lastPaintedTick = t;
            this.lastPaintedPitch = p;
            changed = true;

            const now = Date.now();
            if (now - this.lastPreviewTime > 50 && document.getElementById('preview-audio')?.checked) {
                 this.synth?.triggerAttackRelease(Tone.Frequency(p, "midi"), "16n", Tone.now(), this.brushColor);
                 this.lastPreviewTime = now;
            }
        }
        
        if (changed) {
            this.updateNoteMap();
            this.render();
        }
    }

    paintNoteSimple(t, p) {
        this.maskMap.delete(`${t},${p}`);
        const existingIdx = this.notes.findIndex(n => n.start === t && n.pitch === p);
        let changed = false;
        if (existingIdx !== -1) {
            this.notes.splice(existingIdx, 1);
            changed = true;
        }

        if (this.brushColor > 0.01) {
            this.notes.push({ pitch: p, start: t, duration: 1, velocity: this.brushColor });
            changed = true;
        }
        
        if (changed) this.updateNoteMap();
        this.render();
    }

    saveState() {
        // Deep copy notes and mask
        const state = {
            notes: JSON.parse(JSON.stringify(this.notes)),
            mask: new Set(this.maskMap.keys())
        };
        
        // Remove future history if we are in middle
        if (this.historyIndex < this.history.length - 1) {
            this.history = this.history.slice(0, this.historyIndex + 1);
        }
        
        this.history.push(state);
        if (this.history.length > this.maxHistory) {
            this.history.shift();
        } else {
            this.historyIndex++;
        }
        // Update UI buttons if they exist
        this.updateHistoryButtons();
    }

    undo() {
        if (this.historyIndex > 0) {
            this.historyIndex--;
            this.restoreState(this.history[this.historyIndex]);
        } else if (this.historyIndex === 0) {
            // Undo to initial empty state? 
            // Or just stay at 0? 
            // Usually index points to *current* state.
            // If we saved state BEFORE action, we should step back.
            // My saveState implementation saves *new* state? No, usually we save *before* change.
            // Let's adjust: saveState() should push *current* state.
            // Then we make change.
            // Undo restores the saved state.
        }
        // Wait, standard implementation:
        // 1. Initial state pushed.
        // 2. Action -> push new state.
        // Undo -> index--.
    }

    // Revised saveState/Undo/Redo logic
    // Call saveState() BEFORE making changes (snapshot current) is one way,
    // OR call saveState() AFTER changes (snapshot new).
    // Let's do "Snapshot After".
    // But we need initial state.
    
    // Actually, simpler: 
    // When action starts (mousedown), push copy of CURRENT state to history (as "previous state").
    // Undo -> load that state.
    // Redo -> ...
    
    // Correct Pattern:
    // history = [State0, State1, State2]
    // index = 2 (Current is State2)
    // Undo -> index=1, load State1.
    // Redo -> index=2, load State2.
    // New Action -> index=2, splice history from 3, push State3, index=3.
    
    pushState() {
        const state = {
            notes: JSON.parse(JSON.stringify(this.notes)),
            mask: Array.from(this.maskMap.keys())
        };
        
        // If we undid, discard future
        if (this.historyIndex < this.history.length - 1) {
            this.history = this.history.slice(0, this.historyIndex + 1);
        }
        
        this.history.push(state);
        if (this.history.length > this.maxHistory) this.history.shift();
        else this.historyIndex = this.history.length - 1;
        
        this.updateHistoryButtons();
    }
    
    restoreState(state) {
        if (!state) return;
        this.notes = JSON.parse(JSON.stringify(state.notes));
        this.maskMap.clear();
        state.mask.forEach(k => this.maskMap.set(k, true));
        this.updateNoteMap();
        this.render();
        this.updateHistoryButtons();
    }
    
    triggerUndo() {
        if (this.historyIndex > 0) {
            this.historyIndex--;
            this.restoreState(this.history[this.historyIndex]);
        }
    }
    
    triggerRedo() {
        if (this.historyIndex < this.history.length - 1) {
            this.historyIndex++;
            this.restoreState(this.history[this.historyIndex]);
        }
    }
    
    updateHistoryButtons() {
        const undoBtn = document.getElementById('undo-btn');
        const redoBtn = document.getElementById('redo-btn');
        if (undoBtn) undoBtn.disabled = this.historyIndex <= 0;
        if (redoBtn) redoBtn.disabled = this.historyIndex >= this.history.length - 1;
    }

    setNotes(notes) {
        // Push state before external setting?
        // Usually generation result.
        this.pushState();
        this.notes = [...notes];
        this.updateNoteMap();
        this.maskMap.clear();
        this.render();
        this.pushState(); // Push new state
    }
    
    setConditionRegion(start, end) {
        this.conditionRegion = { start, end };
        this.render();
    }
    
    clearConditionRegion() {
        this.conditionRegion = null;
        this.render();
    }
    
    updateNoteMap() {
        this.noteMap.clear();
        for(const n of this.notes) {
            const tick = Math.round(n.start);
            if(!this.noteMap.has(tick)) this.noteMap.set(tick, []);
            this.noteMap.get(tick).push(n);
        }
    }

    clear() {
        this.notes = [];
        this.noteMap.clear();
        this.maskMap.clear();
        this.conditionRegion = null;
        this.render();
    }
    
    togglePlayback() {
        if (this.isPlaying) this.pausePlayback();
        else this.startPlayback();
    }
    
    async startPlayback() {
        if (!window.Tone) return;
        await Tone.start();
        Tone.Transport.cancel();
        
        this.playbackRepeaterId = Tone.Transport.scheduleRepeat((time) => {
            const ppq = Tone.Transport.PPQ;
            const ticksPer16th = ppq / 4;
            const currentTick = Math.round(Tone.Transport.ticks / ticksPer16th);
            
            if (this.noteMap.has(currentTick)) {
                const notesToPlay = this.noteMap.get(currentTick);
                notesToPlay.forEach(n => {
                    this.synth.triggerAttackRelease(
                        Tone.Frequency(n.pitch, "midi"), 
                        n.duration * Tone.Time("16n").toSeconds(), 
                        time, 
                        n.velocity
                    );
                });
            }
        }, "16n");

        if (this.notes.length > 0) {
            const maxStep = Math.max(...this.notes.map(n => n.start + n.duration));
            const endBar = Math.ceil(maxStep / 16);
            Tone.Transport.loopEnd = `${Math.max(endBar, 1)}:0:0`;
            Tone.Transport.loop = true;
        }
        
        Tone.Transport.start();
        this.isPlaying = true;
        this.render();
        
        const playBtn = document.getElementById('play-btn');
        if(playBtn) playBtn.textContent = '⏸';
    }
    
    pausePlayback() {
        Tone.Transport.pause();
        this.synth?.releaseAll(); 
        this.isPlaying = false;
        const playBtn = document.getElementById('play-btn');
        if(playBtn) playBtn.textContent = '▶';
        this.render();
    }
    
    stopPlayback() {
        Tone.Transport.stop();
        Tone.Transport.cancel();
        this.synth?.releaseAll(); 
        this.isPlaying = false;
        const playBtn = document.getElementById('play-btn');
        if(playBtn) playBtn.textContent = '▶';
        this.render();
    }
}
