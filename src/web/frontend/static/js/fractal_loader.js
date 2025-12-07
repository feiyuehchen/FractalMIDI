// Loading Screen Fractal Animation
class FractalLoader {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width;
        this.height = this.canvas.height;
        this.time = 0;
        this.running = true;
        
        // Julia set params
        this.cRe = -0.7;
        this.cIm = 0.27015;
        
        this.animate();
    }
    
    stop() {
        this.running = false;
    }
    
    animate() {
        if (!this.running) return;
        
        this.time += 0.01;
        
        // Animate parameters
        this.cRe = -0.7 + 0.1 * Math.sin(this.time);
        this.cIm = 0.27015 + 0.1 * Math.cos(this.time * 0.7);
        
        this.draw();
        requestAnimationFrame(() => this.animate());
    }
    
    draw() {
        const w = this.width;
        const h = this.height;
        const maxIter = 30;
        
        const imgData = this.ctx.createImageData(w, h);
        const data = imgData.data;
        
        for (let x = 0; x < w; x++) {
            for (let y = 0; y < h; y++) {
                let zRe = 1.5 * (x - w/2) / (0.5 * w);
                let zIm = (y - h/2) / (0.5 * h);
                
                let i;
                for (i = 0; i < maxIter; i++) {
                    const oldRe = zRe;
                    const oldIm = zIm;
                    
                    zRe = oldRe * oldRe - oldIm * oldIm + this.cRe;
                    zIm = 2 * oldRe * oldIm + this.cIm;
                    
                    if (zRe * zRe + zIm * zIm > 4) break;
                }
                
                const pixelIndex = (y * w + x) * 4;
                
                if (i === maxIter) {
                    // Inside set - black
                    data[pixelIndex] = 0;
                    data[pixelIndex + 1] = 0;
                    data[pixelIndex + 2] = 0;
                    data[pixelIndex + 3] = 255;
                } else {
                    // Outside - color based on iterations
                    const t = i / maxIter;
                    // Cyan to Purple gradient
                    data[pixelIndex] = Math.floor(t * 100);     // R
                    data[pixelIndex + 1] = Math.floor(t * 243); // G
                    data[pixelIndex + 2] = Math.floor(255);     // B
                    data[pixelIndex + 3] = 255;
                }
            }
        }
        
        this.ctx.putImageData(imgData, 0, 0);
    }
}

// Initialize when DOM loaded
document.addEventListener('DOMContentLoaded', () => {
    if(document.getElementById('fractal-canvas')) {
        window.fractalLoader = new FractalLoader('fractal-canvas');
    }
});
