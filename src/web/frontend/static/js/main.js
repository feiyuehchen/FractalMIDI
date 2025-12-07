// Main Application Logic
const API_BASE = window.location.origin;
let currentMode = 'unconditional'; // 'unconditional', 'conditional', 'inpainting'
let currentJobId = null;
let allCheckpoints = []; // Store all checkpoints locally for filtering

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    const loader = document.getElementById('loading-screen');
    if (loader) {
        loader.style.display = 'flex';
        setTimeout(() => {
            if(loader.style.display !== 'none') {
                loader.style.display = 'none';
                console.log("Forced loading screen hide due to timeout");
            }
        }, 5000);
    }

    try {
        initEditor();
        setupEventListeners();
        setupEditorUI();
        
        // Parallel loading
        Promise.all([
            loadCheckpoints(),
            loadExamples()
        ]).then(() => {
            updateUI();
            showLoading(false);
        }).catch(e => {
            console.warn("Partial loading error:", e);
            showLoading(false);
        });
    } catch (e) {
        console.error("Initialization failed:", e);
        showLoading(false);
    }
});

function initEditor() {
    if (document.getElementById('editor-canvas')) {
        window.editor = new EditorCanvas('editor-canvas', 'canvas-container');
    }
}

// Setup all event listeners
function setupEventListeners() {
    // Generator type toggle
    document.querySelectorAll('input[name="generator"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            const scanOrderGroup = document.getElementById('scan-order-group');
            if(scanOrderGroup) scanOrderGroup.style.display = e.target.value === 'ar' ? 'block' : 'none';
            updateCheckpointDropdown();
        });
    });
    
    // Scan order toggle (New)
    document.querySelectorAll('input[name="scan-order"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            updateCheckpointDropdown();
        });
    });
    
    // Mode tabs (Generation Mode)
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const mode = e.target.dataset.mode;
            switchMode(mode);
        });
    });
    
    // Range inputs
    const ranges = ['uncond-length', 'cond-length', 'cond-total-length', 'temperature', 'cfg'];
    ranges.forEach(id => setupRangeInput(id));
    
    // Buttons
    const loadBtn = document.getElementById('load-checkpoint-btn');
    if (loadBtn) loadBtn.addEventListener('click', loadSelectedCheckpoint);
    
    const checkpointSelect = document.getElementById('checkpoint-select');
    if (checkpointSelect) checkpointSelect.addEventListener('change', loadSelectedCheckpoint);

    const generateBtn = document.getElementById('generate-btn');
    if (generateBtn) generateBtn.addEventListener('click', startGeneration);
    
    const regenBtn = document.getElementById('regenerate-btn');
    if (regenBtn) regenBtn.addEventListener('click', startRegeneration);
    
    // Example Selects - Load into Canvas
    const condSelect = document.getElementById('cond-example-select');
    if (condSelect) condSelect.addEventListener('change', loadExampleToCanvas);
    
    const inpaintSelect = document.getElementById('inpaint-example-select');
    if (inpaintSelect) inpaintSelect.addEventListener('change', loadExampleToCanvas);
}

async function loadExampleToCanvas(e) {
    const exampleId = e.target.value;
    if (!exampleId || !window.editor) return;
    
    try {
        showLoading(true);
        const response = await fetch(`${API_BASE}/api/examples/${exampleId}/notes`);
        if (!response.ok) throw new Error("Failed to fetch example notes");
        const data = await response.json();
        
        window.editor.setNotes(data.notes);
        window.editor.fitView();
        
        // If conditional mode, set condition region highlight based on default or existing slider
        if (currentMode === 'conditional') {
            const len = parseInt(document.getElementById('cond-length')?.value || 32);
            window.editor.setConditionRegion(0, len);
        } else if (currentMode === 'inpainting') {
            window.editor.clearConditionRegion(); // Inpainting uses Mask (Eraser)
        }
        
    } catch (error) {
        console.error("Error loading example notes:", error);
        showError("Could not load example notes to canvas");
    } finally {
        showLoading(false);
    }
}

function setupEditorUI() {
    if (!window.editor) return;

    // View/Edit Mode Toggle
    const modeOptions = document.querySelectorAll('.mode-option');
    const propertyBar = document.getElementById('property-bar');
    const editActions = document.getElementById('edit-actions');
    
    modeOptions.forEach(opt => {
        opt.addEventListener('click', () => {
            // Toggle active class
            modeOptions.forEach(o => o.classList.remove('active'));
            opt.classList.add('active');
            
            const mode = opt.dataset.viewMode;
            window.editor.mode = mode;
            
            if (mode === 'edit') {
                propertyBar.classList.remove('hidden');
                if(editActions) editActions.style.display = 'block';
            } else {
                propertyBar.classList.add('hidden');
                if(editActions) editActions.style.display = 'none';
            }
        });
    });

    // Transport Controls
    document.getElementById('play-btn')?.addEventListener('click', () => window.editor.togglePlayback());
    document.getElementById('stop-btn')?.addEventListener('click', () => window.editor.stopPlayback());
    
    // Tempo
    const tempoInput = document.getElementById('tempo-input');
    if (tempoInput) {
        tempoInput.addEventListener('change', (e) => {
            let bpm = parseInt(e.target.value);
            if (bpm < 40) bpm = 40;
            if (bpm > 300) bpm = 300;
            e.target.value = bpm;
            window.editor.setBPM(bpm);
        });
    }

    // Zoom Controls
    document.getElementById('zoom-in-btn')?.addEventListener('click', () => {
        window.editor.zoomX *= 1.2; 
        window.editor.render();
    });
    document.getElementById('zoom-out-btn')?.addEventListener('click', () => {
        window.editor.zoomX *= 0.8;
        window.editor.render();
    });
    
    const resetBtn = document.getElementById('reset-view-btn');
    if (resetBtn) {
        resetBtn.textContent = "FIT"; // Rename to FIT
        resetBtn.addEventListener('click', () => {
            window.editor.fitView();
        });
    }

    // Tools (Pen/Eraser)
    const tools = document.querySelectorAll('.tool-btn-large[data-tool]');
    tools.forEach(btn => {
        btn.addEventListener('click', () => {
            tools.forEach(t => t.classList.remove('active'));
            btn.classList.add('active');
            window.editor.tool = btn.dataset.tool;
        });
    });

    // Brush Size
    const sizeSlider = document.getElementById('brush-size');
    if (sizeSlider) {
        sizeSlider.addEventListener('input', (e) => {
            window.editor.brushSize = parseInt(e.target.value);
            const preview = document.querySelector('.size-preview');
            if(preview) preview.textContent = `${e.target.value}px`;
        });
    }

    // Eraser Mode Toggle
    const eraserBtn = document.getElementById('tool-eraser');
    if (eraserBtn) {
        // Create sub-options for eraser
        // This assumes HTML structure allows it, or we inject it.
        // For now, let's just toggle logic if user clicks eraser again? 
        // Or better, add a separate control in HTML.
    }
    
    // Undo/Redo Shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.metaKey) {
            if (e.key === 'z') {
                e.preventDefault();
                window.editor?.triggerUndo();
            } else if (e.key === 'y') {
                e.preventDefault();
                window.editor?.triggerRedo();
            }
        }
    });

    // Undo/Redo Buttons (if added to HTML)
    document.getElementById('undo-btn')?.addEventListener('click', () => window.editor?.triggerUndo());
    document.getElementById('redo-btn')?.addEventListener('click', () => window.editor?.triggerRedo());

    // Eraser Option in Tool Palette
    const eraserModeSelect = document.getElementById('eraser-mode-select');
    if(eraserModeSelect) {
        eraserModeSelect.addEventListener('change', (e) => {
            if(window.editor) window.editor.eraserMode = e.target.value;
        });
    } else {
        // Fallback: Inject if not present in HTML (hacky but works for immediate fix)
    }

    // Color Palette
    const paletteContainer = document.getElementById('color-palette');
    if (paletteContainer) {
        // Create swatches
        // 5 colors + Black
        const colors = [
             { vel: 0.2, color: 'rgb(0, 128, 255)' },
             { vel: 0.4, color: 'rgb(0, 255, 128)' },
             { vel: 0.6, color: 'rgb(255, 255, 0)' },
             { vel: 0.8, color: 'rgb(255, 128, 0)' },
             { vel: 1.0, color: 'rgb(255, 0, 0)' },
             { vel: 0.01, color: '#000', border: true } // Black (Eraser/Silence)
        ];
        
        colors.forEach(c => {
            const swatch = document.createElement('div');
            swatch.className = 'color-swatch';
            swatch.style.backgroundColor = c.color;
            if (c.border) swatch.style.borderColor = '#555';
            
            swatch.addEventListener('click', () => {
                document.querySelectorAll('.color-swatch').forEach(s => s.classList.remove('active'));
                swatch.classList.add('active');
                window.editor.brushColor = c.vel;
            });
            
            paletteContainer.appendChild(swatch);
        });
        
        // Select default (Orange 0.8)
        paletteContainer.children[3].classList.add('active');
    }
    
    // Condition Length Slider Listener
    const condLenInput = document.getElementById('cond-length');
    if (condLenInput) {
        condLenInput.addEventListener('input', (e) => {
            if (currentMode === 'conditional' && window.editor) {
                // Keep current start, update end
                const currentStart = window.editor.conditionRegion ? window.editor.conditionRegion.start : 0;
                const len = parseInt(e.target.value);
                window.editor.setConditionRegion(currentStart, currentStart + len);
            }
        });
    }
}

function setupRangeInput(id) {
    const input = document.getElementById(id);
    const valueSpan = document.getElementById(`${id}-value`);
    if(input && valueSpan) {
        input.addEventListener('input', () => {
            valueSpan.textContent = input.value;
        });
    }
}

function switchMode(mode) {
    currentMode = mode;
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });
    document.querySelectorAll('.mode-params').forEach(content => {
        content.classList.toggle('active', content.id === `mode-${mode}`);
    });
    
    // Update Canvas Overlay
    if (window.editor) {
        if (mode === 'conditional') {
            const len = parseInt(document.getElementById('cond-length')?.value || 32);
            window.editor.setConditionRegion(0, len);
        } else {
            window.editor.clearConditionRegion();
        }
    }
}

// Load available checkpoints
async function loadCheckpoints() {
    try {
        const response = await fetch(`${API_BASE}/api/models/list`);
        const data = await response.json();
        
        allCheckpoints = data.checkpoints || [];
        
        if (allCheckpoints.length === 0) {
            const select = document.getElementById('checkpoint-select');
            select.innerHTML = '<option value="">No checkpoints found</option>';
            return;
        }

        // Initial update
        updateCheckpointDropdown();
        
        // Check current model status and try to auto-select
        try {
             const statusResponse = await fetch(`${API_BASE}/api/models/info`);
             const statusData = await statusResponse.json();
             if (statusData.loaded) {
                 const select = document.getElementById('checkpoint-select');
                 if (select.querySelector(`option[value="${statusData.checkpoint}"]`)) {
                     select.value = statusData.checkpoint;
                     updateModelInfo(statusData);
                 }
             }
        } catch(e) {}
        
    } catch (error) {
        console.error('Error loading checkpoints:', error);
        showError('Failed to load checkpoints');
    }
}

function updateCheckpointDropdown() {
    const select = document.getElementById('checkpoint-select');
    if (!select) return;
    
    const generatorType = document.querySelector('input[name="generator"]:checked')?.value || 'mar';
    const scanOrder = document.querySelector('input[name="scan-order"]:checked')?.value || 'row_major';
    
    // Filter checkpoints
    const filtered = allCheckpoints.filter(ckpt => {
        // If no metadata, assume MAR (backward compat)
        const types = ckpt.generator_types;
        
        let matchType = false;
        if (generatorType === 'mar') {
            // Match if explicit MAR or unknown
            if (!types || types.length === 0) return true; 
            matchType = types.every(t => t === 'mar');
        } else { // AR
            if (!types || types.length === 0) return false; 
            matchType = types.every(t => t === 'ar');
        }
        
        if (!matchType) return false;
        
        // Check scan order (only relevant for AR)
        if (generatorType === 'ar') {
            const ckptOrder = ckpt.scan_order || 'row_major'; // Default
            return ckptOrder === scanOrder;
        }
        
        return true;
    });
    
    // Populate
    select.innerHTML = '';
    if (filtered.length === 0) {
        select.innerHTML = '<option value="">No matching checkpoints</option>';
        return;
    }
    
    filtered.forEach(ckpt => {
        const option = document.createElement('option');
        option.value = ckpt.name;
        const size = ckpt.file_size_mb ? ckpt.file_size_mb.toFixed(1) : "0.0";
        let label = `${ckpt.name}`;
        if (ckpt.step) label += ` (Step ${ckpt.step})`;
        // label += ` - ${size}MB`;
        option.textContent = label;
        select.appendChild(option);
    });
    
    // Auto select first if nothing selected (or if current selection is invalid)
    // We'll just select first one for now to keep it simple
    if (filtered.length > 0) {
        select.value = filtered[0].name;
        // Optionally trigger load immediately? 
        // Better to let user click "Load" or select, but app auto-loads on init maybe?
        // Current logic has explicit load via change event or load button (which doesn't exist in new UI?)
        // Ah, there is no "Load" button in index.html anymore? 
        // Checking setupEventListeners... 
        // if (checkpointSelect) checkpointSelect.addEventListener('change', loadSelectedCheckpoint);
        // So changing select triggers load.
        // So we should trigger load here.
        loadSelectedCheckpoint();
    }
}

async function loadSelectedCheckpoint() {
    const select = document.getElementById('checkpoint-select');
    const checkpointName = select.value;
    if (!checkpointName) return;
    
    showLoading(true);
    try {
        const response = await fetch(`${API_BASE}/api/models/load`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ checkpoint_name: checkpointName })
        });
        const data = await response.json();
        if (data.status === 'success') {
            updateModelInfo(data.model_info);
            showSuccess('Model loaded successfully');
        } else {
            throw new Error(data.message || 'Failed to load model');
        }
    } catch (error) {
        // Don't show alert on auto-load failure, just log
        console.error(error.message);
        // showError(error.message); 
    } finally {
        setTimeout(() => showLoading(false), 500);
    }
}

function updateModelInfo(info) {
    const infoBox = document.getElementById('model-info');
    const infoText = document.getElementById('model-info-text');
    if (infoBox && infoText) {
        infoBox.style.display = 'block';
        infoText.textContent = `${info.checkpoint} (${info.parameters_millions?.toFixed(1)}M params)`;
    }
}

async function loadExamples() {
    try {
        const response = await fetch(`${API_BASE}/api/examples/list?limit=50`);
        const data = await response.json();
        const condSelect = document.getElementById('cond-example-select');
        const inpaintSelect = document.getElementById('inpaint-example-select');
        
        data.examples.forEach(ex => {
            const text = `${ex.name} (${ex.duration_seconds.toFixed(1)}s)`;
            const opt1 = document.createElement('option'); opt1.value = ex.id; opt1.textContent = text;
            condSelect.appendChild(opt1);
            const opt2 = document.createElement('option'); opt2.value = ex.id; opt2.textContent = text;
            inpaintSelect.appendChild(opt2);
        });
    } catch (error) {
        console.error('Error loading examples:', error);
    }
}

// Start generation (Standard)
async function startGeneration() {
    const request = buildRequest();
    if (!request) return;

    if (window.editor) {
        // Only clear if Unconditional mode
        if (currentMode === 'unconditional') {
            window.editor.clear();
        } else {
            // For conditional/inpainting, we keep the canvas notes
            // But we might want to clear the "generated" part if we can distinguish it?
            // No, keeping context is better.
        }
    }
    
    await runGeneration(request);
}

// Start Regeneration (Inpainting with custom mask)
async function startRegeneration() {
    if (!window.editor) return;

    const maskPoints = [];
    for (const key of window.editor.maskMap.keys()) {
        const [t, p] = key.split(',').map(Number);
        maskPoints.push([t, p]);
    }
    
    if (maskPoints.length === 0) {
        showError("No mask defined. Use the Eraser tool to mask areas.");
        return;
    }
    
    const request = {
        mode: 'inpainting_custom', // Custom mode for backend
        generator_type: document.querySelector('input[name="generator"]:checked')?.value || 'mar',
        scan_order: document.querySelector('input[name="scan-order"]:checked')?.value || 'row_major', // Add scan order
        temperature: parseFloat(document.getElementById('temperature').value),
        cfg: parseFloat(document.getElementById('cfg').value),
        create_gif: document.getElementById('create-gif')?.checked || false,
        
        // Custom Data
        user_notes: window.editor.notes,
        mask_points: maskPoints,
        length: 256 // Or derive from max(notes)
    };
    
    await runGeneration(request);
}

function buildRequest() {
    const generatorTypeEl = document.querySelector('input[name="generator"]:checked');
    if (!generatorTypeEl) { showError("Generator type not selected"); return null; }
    
    let request = {
        mode: currentMode,
        generator_type: generatorTypeEl.value,
        scan_order: document.querySelector('input[name="scan-order"]:checked')?.value || 'row_major',
        temperature: parseFloat(document.getElementById('temperature').value),
        cfg: parseFloat(document.getElementById('cfg').value),
        create_gif: document.getElementById('create-gif')?.checked || false,
        show_progress: true,
        show_grid: document.getElementById('show-grid')?.checked || false
    };

    if (currentMode === 'unconditional') {
        request.length = parseInt(document.getElementById('uncond-length').value);
    } else if (currentMode === 'conditional') {
        const exampleId = document.getElementById('cond-example-select').value;
        if (!exampleId) { showError('Please select an example'); return null; }
        request.condition_example_id = exampleId;
        
        // Use the condition range from canvas if available
        if (window.editor && window.editor.conditionRegion) {
             const region = window.editor.conditionRegion;
             request.condition_start = region.start;
             request.condition_end = region.end;
             request.condition_length = region.end - region.start;
        } else {
             // Fallback
             request.condition_start = 0;
             request.condition_length = parseInt(document.getElementById('cond-length').value);
             request.condition_end = request.condition_length;
        }
        
        request.length = parseInt(document.getElementById('cond-total-length')?.value || 256);
    } else if (currentMode === 'inpainting') {
        const exampleId = document.getElementById('inpaint-example-select').value;
        if (!exampleId) { showError('Please select an example'); return null; }
        request.inpaint_example_id = exampleId;
        request.length = 256;
    }
    return request;
}

async function runGeneration(request) {
    const btn = document.getElementById('generate-btn');
    const regenBtn = document.getElementById('regenerate-btn');
    
    if(btn) btn.disabled = true;
    if(regenBtn) regenBtn.disabled = true;
    
    // Lock UI
    const modeOptions = document.querySelectorAll('.mode-option');
    modeOptions.forEach(o => o.style.pointerEvents = 'none');
    
    // Force stop playback
    if(window.editor) window.editor.stopPlayback();
    
    const progressContainer = document.getElementById('progress-container');
    if(progressContainer) progressContainer.style.display = 'block';
    
    try {
        await generateWithWebSocket(request);
    } catch (error) {
        console.error('Generation error:', error);
        showError('Generation failed: ' + error.message);
    } finally {
        if(btn) btn.disabled = false;
        if(regenBtn) regenBtn.disabled = false;
        modeOptions.forEach(o => o.style.pointerEvents = 'auto');
    }
}

async function generateWithWebSocket(request) {
    return new Promise((resolve, reject) => {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(`${protocol}//${window.location.host}/ws/generate`);
        
        ws.onopen = () => {
            ws.send(JSON.stringify(request));
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.status === 'error') {
                reject(new Error(data.message));
                return;
            }
            
            if (data.progress !== undefined) {
                updateProgress(data.progress, data.message || 'Generating...');
            }
            
            // Handle realtime notes
            if (data.type === 'new_notes' && window.editor) {
                 window.editor.setNotes(data.notes);
            }
            
            // Handle condition influence
            if (data.type === 'condition_influence' && window.editor) {
                window.editor.setInfluence(data.level, data.influence);
            }
            
            if (data.status === 'completed') {
                displayResults(data);
                resolve();
                ws.close();
            }
        };
        
        ws.onerror = (error) => {
            console.error("WebSocket error", error);
            reject(error);
        };
        
        ws.onclose = () => {
            const pc = document.getElementById('progress-container');
            if(pc) pc.style.display = 'none';
        };
    });
}

function updateProgress(progress, message) {
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    if(progressFill) progressFill.style.width = `${progress * 100}%`;
    if(progressText) progressText.textContent = message;
}

function displayResults(data) {
    const resultsPanel = document.getElementById('results-panel');
    const resultImage = document.getElementById('result-image');
    const downloadMidi = document.getElementById('download-midi-btn');
    const downloadImage = document.getElementById('download-image-btn');
    const downloadGif = document.getElementById('download-gif-btn');
    
    if (resultImage && data.image_url) {
        resultImage.src = `${API_BASE}${data.image_url}`;
        resultImage.style.display = 'block';
    }
    
    if (downloadMidi) downloadMidi.href = `${API_BASE}${data.midi_url}`;
    if (downloadImage) downloadImage.href = `${API_BASE}${data.image_url}`;
    
    if (data.gif_url && downloadGif) {
        downloadGif.href = `${API_BASE}${data.gif_url}`;
        downloadGif.style.display = 'inline-block';
    } else if (downloadGif) {
        downloadGif.style.display = 'none';
    }
    
    if (resultsPanel) {
        resultsPanel.style.display = 'block';
    }
}

function showLoading(show) {
    const loader = document.getElementById('loading-screen');
    if(loader) loader.style.display = show ? 'flex' : 'none';
}

function showError(message) {
    alert('Error: ' + message);
}

function showSuccess(message) {
    console.log('Success:', message);
}

function updateUI() {
    // Helper to refresh UI state if needed
}
