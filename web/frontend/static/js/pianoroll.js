// Piano Roll Canvas Renderer

class PianoRollRenderer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.pianoRoll = null;
        this.zoom = 1.0;
        this.offsetX = 0;
        this.offsetY = 0;
        this.isDragging = false;
        this.lastX = 0;
        this.lastY = 0;
        
        // Colormap (matching Python visualizer)
        this.createColormap();
        
        // Setup event listeners
        this.setupEventListeners();
    }
    
    createColormap() {
        // Create colormap matching Python VELOCITY_CMAP
        this.colormap = [];
        const colors = [
            [0, 0, 0],       // Black (silence)
            [0, 0, 128],     // Dark blue
            [0, 128, 255],   // Blue
            [0, 255, 128],   // Cyan
            [255, 255, 0],   // Yellow
            [255, 128, 0],   // Orange
            [255, 0, 0]      // Red (loud)
        ];
        
        // Interpolate colors
        const steps = 256;
        for (let i = 0; i < steps; i++) {
            const t = i / (steps - 1);
            const segmentIndex = Math.floor(t * (colors.length - 1));
            const segmentT = (t * (colors.length - 1)) - segmentIndex;
            
            const c1 = colors[Math.min(segmentIndex, colors.length - 1)];
            const c2 = colors[Math.min(segmentIndex + 1, colors.length - 1)];
            
            const r = Math.round(c1[0] * (1 - segmentT) + c2[0] * segmentT);
            const g = Math.round(c1[1] * (1 - segmentT) + c2[1] * segmentT);
            const b = Math.round(c1[2] * (1 - segmentT) + c2[2] * segmentT);
            
            this.colormap.push(`rgb(${r},${g},${b})`);
        }
    }
    
    setupEventListeners() {
        // Mouse events for panning
        this.canvas.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.lastX = e.clientX;
            this.lastY = e.clientY;
        });
        
        this.canvas.addEventListener('mousemove', (e) => {
            if (this.isDragging) {
                const dx = e.clientX - this.lastX;
                const dy = e.clientY - this.lastY;
                this.offsetX += dx;
                this.offsetY += dy;
                this.lastX = e.clientX;
                this.lastY = e.clientY;
                this.render();
            }
        });
        
        this.canvas.addEventListener('mouseup', () => {
            this.isDragging = false;
        });
        
        this.canvas.addEventListener('mouseleave', () => {
            this.isDragging = false;
        });
        
        // Wheel for zooming
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            this.zoom *= delta;
            this.zoom = Math.max(0.5, Math.min(5.0, this.zoom));
            this.render();
        });
    }
    
    setPianoRoll(pianoRoll) {
        // pianoRoll should be 2D array [128][T]
        this.pianoRoll = pianoRoll;
        this.render();
    }
    
    render() {
        if (!this.pianoRoll) {
            this.ctx.fillStyle = '#0a0a0a';
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            return;
        }
        
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        // Clear canvas
        this.ctx.fillStyle = '#0a0a0a';
        this.ctx.fillRect(0, 0, width, height);
        
        const numPitches = this.pianoRoll.length;
        const numTimeSteps = this.pianoRoll[0].length;
        
        const cellWidth = (width / numTimeSteps) * this.zoom;
        const cellHeight = (height / numPitches) * this.zoom;
        
        // Render piano roll
        for (let pitch = 0; pitch < numPitches; pitch++) {
            for (let time = 0; time < numTimeSteps; time++) {
                const velocity = this.pianoRoll[pitch][time];
                
                if (velocity > 0) {
                    const colorIndex = Math.floor(velocity * 255);
                    this.ctx.fillStyle = this.colormap[colorIndex];
                    
                    const x = time * cellWidth + this.offsetX;
                    const y = (numPitches - pitch - 1) * cellHeight + this.offsetY;
                    
                    // Only render if visible
                    if (x + cellWidth >= 0 && x < width && y + cellHeight >= 0 && y < height) {
                        this.ctx.fillRect(x, y, Math.ceil(cellWidth), Math.ceil(cellHeight));
                    }
                }
            }
        }
    }
    
    zoomIn() {
        this.zoom *= 1.2;
        this.zoom = Math.min(5.0, this.zoom);
        this.render();
    }
    
    zoomOut() {
        this.zoom *= 0.8;
        this.zoom = Math.max(0.5, this.zoom);
        this.render();
    }
    
    resetView() {
        this.zoom = 1.0;
        this.offsetX = 0;
        this.offsetY = 0;
        this.render();
    }
    
    clear() {
        this.pianoRoll = null;
        this.render();
    }
}

// Global instance
let pianoRollRenderer = null;

function initPianoRollRenderer() {
    pianoRollRenderer = new PianoRollRenderer('piano-roll-canvas');
}

