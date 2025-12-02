// Immersive Piano Roll Renderer with Growth & Particles

class ImmersivePianoRoll {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        
        // State
        this.notes = []; // {pitch, start, end, velocity, spawnTime, growth: 0->1}
        this.particles = []; // {x, y, vx, vy, life, color}
        
        // Viewport
        this.zoom = 1.0;
        this.offsetX = 0;
        this.offsetY = 0;
        
        // Config
        this.gridColor = 'rgba(255, 255, 255, 0.05)';
        this.noteHeight = 4; // Base height per pitch
        this.ticksPerPixel = 1;
        
        // Color Palette (Deep Space Neon)
        this.colors = [
            [0, 243, 255],   // Cyan
            [0, 255, 157],   // Green
            [176, 38, 255],  // Purple
            [255, 0, 128],   // Pink
            [255, 200, 0]    // Gold
        ];

        // Bindings
        this.resize();
        window.addEventListener('resize', () => this.resize());
        this.setupInteraction();
        
        // Start loop
        this.lastTime = performance.now();
        this.animate = this.animate.bind(this);
        requestAnimationFrame(this.animate);
    }

    resize() {
        // Use parent container size instead of clientWidth/Height directly if canvas is styled 100%
        // But wrapper logic is safer
        const wrapper = this.canvas.parentElement;
        if(wrapper) {
            this.canvas.width = wrapper.clientWidth;
            this.canvas.height = wrapper.clientHeight;
        } else {
            this.canvas.width = window.innerWidth;
            this.canvas.height = window.innerHeight;
        }
        
        // Trigger re-render
        this.render();
    }
    
    // Add new notes (e.g. from WebSocket)
    addNotes(newNotes) {
        // newNotes: list of {pitch, start, duration, velocity}
        const now = performance.now();
        const isBulkLoad = newNotes.length > 50; // If adding many notes, reduce effects
        
        newNotes.forEach(n => {
            this.notes.push({
                ...n,
                end: n.start + n.duration,
                spawnTime: now,
                growth: 0,
                popped: false
            });
            
            // Spawn particles at note start (only if not bulk loading to save performance)
            if (!isBulkLoad) {
                this.spawnParticles(n);
            }
        });
    }
    
    spawnParticles(note) {
        if (this.particles.length > 500) return; // Hard limit on particles
        
        // Convert time/pitch to screen coords roughly (re-calculated in render usually, 
        // but we need spawn position now or just store logical pos)
        // Let's store logical pos and render particles relative to viewport
        const cnt = 3 + Math.floor(note.velocity * 5); // Reduced count
        for(let i=0; i<cnt; i++) {
            const angle = Math.random() * Math.PI * 2;
            const speed = Math.random() * 2 + 1;
            this.particles.push({
                // Logical coordinates
                t: note.start + Math.random() * note.duration * 0.2, // Slightly spread
                p: note.pitch + Math.random() * 0.8,
                vx: Math.cos(angle) * speed * 0.1, // Velocity in tick/pitch space
                vy: Math.sin(angle) * speed * 0.1,
                life: 1.0,
                color: this.getColor(note.velocity)
            });
        }
    }

    getColor(velocity) {
        // Map velocity 0-1 to palette
        const idx = Math.floor(velocity * (this.colors.length - 1));
        const c = this.colors[idx];
        return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
    }
    
    // Easing: Elastic Out
    easeElastic(x) {
        const c4 = (2 * Math.PI) / 3;
        return x === 0 ? 0 : x === 1 ? 1 : Math.pow(2, -10 * x) * Math.sin((x * 10 - 0.75) * c4) + 1;
    }

    update(dt) {
        const now = performance.now();
        
        // Update notes growth
        this.notes.forEach(n => {
            const age = (now - n.spawnTime) / 1000; // seconds
            const duration = 0.8; // growth duration
            n.growth = Math.min(1.0, age / duration);
        });
        
        // Update particles
        for(let i = this.particles.length - 1; i >= 0; i--) {
            const p = this.particles[i];
            p.t += p.vx;
            p.p += p.vy;
            p.life -= dt * 2.0; // Fade out speed
            if(p.life <= 0) this.particles.splice(i, 1);
        }
    }
    
    animate(time) {
        const dt = (time - this.lastTime) / 1000;
        this.lastTime = time;
        
        this.update(dt);
        this.render();
        
        requestAnimationFrame(this.animate);
    }

    setupInteraction() {
        let isDragging = false;
        let lastX = 0;
        let lastY = 0;

        this.canvas.addEventListener('mousedown', e => {
            isDragging = true;
            lastX = e.clientX;
            lastY = e.clientY;
        });

        window.addEventListener('mouseup', () => {
            isDragging = false;
        });

        window.addEventListener('mousemove', e => {
            if (!isDragging) return;
            const dx = e.clientX - lastX;
            const dy = e.clientY - lastY;
            
            this.offsetX += dx;
            this.offsetY += dy;
            
            lastX = e.clientX;
            lastY = e.clientY;
        });
        
        // Wheel zoom
        this.canvas.addEventListener('wheel', e => {
            e.preventDefault();
            const zoomSpeed = 0.001;
            this.zoom += -e.deltaY * zoomSpeed;
            this.zoom = Math.max(0.1, Math.min(this.zoom, 5.0));
        });
    }

    render() {
        const { width, height } = this.canvas;
        this.ctx.clearRect(0, 0, width, height);
        
        // Background Grid (Pixel art style)
        this.drawGrid();
        
        // Draw "Void" border if width > height to imply infinite space
        
        // Render Notes
        const cellH = (height / 128) * this.zoom;
        const cellW = (width / 256) * this.zoom; // visible window width roughly
        
        // Optimization: Only draw visible notes
        // Skipping strict culling for now for simplicity
        
        this.ctx.globalCompositeOperation = 'screen'; // Neon blend mode
        
        this.notes.forEach(n => {
            const x = (n.start * cellW) + this.offsetX;
            const w = (n.duration * cellW);
            const y = height - ((n.pitch + 1) * cellH) + this.offsetY;
            const h = cellH;
            
            // Growth effect
            const scale = this.easeElastic(n.growth);
            
            const centerX = x + w/2;
            const centerY = y + h/2;
            
            // Draw
            this.ctx.fillStyle = this.getColor(n.velocity);
            
            // Pixel block style instead of rounded rect for "Pixel" theme
            // But keeping a slight glow
            
            const drawW = Math.max(1, w * scale);
            const drawH = Math.max(1, h * scale);
            const drawX = centerX - drawW/2;
            const drawY = centerY - drawH/2;
            
            // Main block
            this.ctx.fillRect(drawX, drawY, drawW, drawH);
            
            // Inner highlight (Pixel accent)
            if (drawH > 4 && drawW > 4) {
                this.ctx.fillStyle = 'rgba(255,255,255,0.5)';
                this.ctx.fillRect(drawX + 2, drawY + 2, 2, 2);
            }
            
            // Glow
            if(n.growth < 0.5) {
                this.ctx.shadowBlur = 20 * (1 - n.growth);
                this.ctx.shadowColor = 'white';
            } else {
                this.ctx.shadowBlur = 10;
                this.ctx.shadowColor = this.getColor(n.velocity);
            }
        });
        
        this.ctx.shadowBlur = 0; // Reset
        
        // Render Particles
        this.particles.forEach(p => {
            const x = (p.t * cellW) + this.offsetX;
            const y = height - ((p.p + 1) * cellH) + this.offsetY;
            
            this.ctx.fillStyle = p.color;
            this.ctx.globalAlpha = p.life;
            
            // Draw as square pixels
            const size = Math.max(1, 3 * p.life);
            this.ctx.fillRect(x, y, size, size);
        });
        this.ctx.globalAlpha = 1.0;
    }

    drawGrid() {
        const { width, height } = this.canvas;
        const cellH = (height / 128) * this.zoom;
        const cellW = (width / 256) * this.zoom;
        
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
        this.ctx.lineWidth = 1;
        
        this.ctx.beginPath();
        // Horizontal lines (Octaves)
        for (let i = 0; i <= 128; i += 12) {
            const y = height - (i * cellH) + this.offsetY;
            if (y >= 0 && y <= height) {
                this.ctx.moveTo(0, y);
                this.ctx.lineTo(width, y);
            }
        }
        
        // Vertical lines (Bars)
        // Assume 4 beats * 4 16th notes = 16 steps per bar
        for (let i = 0; i <= 1000; i += 16) {
             const x = (i * cellW) + this.offsetX;
             if (x >= 0 && x <= width) {
                 this.ctx.moveTo(x, 0);
                 this.ctx.lineTo(x, height);
             }
        }
        this.ctx.stroke();
    }
}

// Initialize
window.addEventListener('DOMContentLoaded', () => {
    if(document.getElementById('piano-roll-canvas')) {
        window.immersivePianoRoll = new ImmersivePianoRoll('piano-roll-canvas');
    }
});
