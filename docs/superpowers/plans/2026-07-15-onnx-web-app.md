# YOLO ONNX Web Application & CI/CD Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a static web application to run real-time fruit detection using ONNX Runtime Web via the browser camera, and set up a CI/CD GitHub Actions workflow to automate deployment to GitHub Pages.

**Architecture:** Build a clean, dark-themed HTML/CSS/JS frontend located inside the `docs/` folder, referencing `onnxruntime-web` from a CDN. Deploy using a GitHub Actions workflow `.github/workflows/deploy.yml` that publishes the `docs/` folder on commits to `main`.

**Tech Stack:** HTML5, CSS3, JavaScript, ONNX Runtime Web, GitHub Actions.

## Global Constraints

*   NO EMOJIS EVER: Do not write or include emojis in the code changes, print messages, comments, configurations, logs, or documentation.
*   Prefix all CLI/shell executions with `rtk` (Rust Token Killer) to minimize context token overhead.

---

### Task 1: Create the Web App UI and Stylesheet

**Files:**
*   Create: `docs/index.html`
*   Create: `docs/styles.css`

**Interfaces:**
*   Consumes: ONNX Runtime Web CDN script.
*   Produces: Static HTML/CSS interface for displaying the webcam feed, control sliders, and detection statistics.

*   [ ] **Step 1: Create index.html**
    Create [docs/index.html](file:///home/tgrwjya/Documents/Uni/Semester%206/AI_ML/CV/Project_Work/Fruits-detection/docs/index.html) with input controls, video preview, overlay canvas, and stats panel:

    ```html
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Real-Time Fruit Detection (YOLO ONNX)</title>
        <link rel="stylesheet" href="styles.css">
        <!-- ONNX Runtime Web CDN -->
        <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
    </head>
    <body>
        <div class="app-container">
            <header>
                <h1>Real-Time Fruit Detection</h1>
                <p>YOLOv8s / YOLO26s Browser Inference</p>
            </header>

            <main class="main-layout">
                <div class="video-section">
                    <div class="video-wrapper">
                        <video id="webcam" autoplay playsinline muted></video>
                        <canvas id="detection-canvas"></canvas>
                        <div id="loading-overlay" class="overlay">
                            <div class="spinner"></div>
                            <p id="loading-text">Loading Model...</p>
                        </div>
                    </div>
                    <div class="controls-panel">
                        <div class="control-group">
                            <label for="model-select">Select Model</label>
                            <select id="model-select">
                                <option value="yolo26s.onnx">YOLO26s (Default)</option>
                                <option value="yolov8s.onnx">YOLOv8s</option>
                            </select>
                        </div>
                        <div class="control-group">
                            <label for="conf-threshold">Confidence: <span id="conf-val">0.40</span></label>
                            <input type="range" id="conf-threshold" min="0.1" max="0.9" step="0.05" value="0.40">
                        </div>
                        <div class="control-group">
                            <label for="iou-threshold">IoU Threshold: <span id="iou-val">0.45</span></label>
                            <input type="range" id="iou-threshold" min="0.1" max="0.9" step="0.05" value="0.45">
                        </div>
                        <div class="button-group">
                            <button id="start-btn" class="btn primary">Start Camera</button>
                            <button id="pause-btn" class="btn secondary" disabled>Pause</button>
                        </div>
                    </div>
                </div>

                <div class="stats-section">
                    <div class="card stat-card">
                        <h3>Inference Performance</h3>
                        <div class="stat-row">
                            <span>Inference Time:</span>
                            <span id="inference-time" class="val">-- ms</span>
                        </div>
                        <div class="stat-row">
                            <span>FPS:</span>
                            <span id="fps-counter" class="val">--</span>
                        </div>
                    </div>

                    <div class="card classes-card">
                        <h3>Detection Counts</h3>
                        <div id="class-counts-container">
                            <!-- Populated dynamically -->
                            <div class="class-row"><span>Apple:</span> <span id="count-Apple">0</span></div>
                            <div class="class-row"><span>Banana:</span> <span id="count-Banana">0</span></div>
                            <div class="class-row"><span>Grape:</span> <span id="count-Grape">0</span></div>
                            <div class="class-row"><span>Orange:</span> <span id="count-Orange">0</span></div>
                            <div class="class-row"><span>Pineapple:</span> <span id="count-Pineapple">0</span></div>
                            <div class="class-row"><span>Watermelon:</span> <span id="count-Watermelon">0</span></div>
                        </div>
                    </div>
                </div>
            </main>
        </div>
        <script src="app.js"></script>
    </body>
    </html>
    ```

*   [ ] **Step 2: Create styles.css**
    Create [docs/styles.css](file:///home/tgrwjya/Documents/Uni/Semester%206/AI_ML/CV/Project_Work/Fruits-detection/docs/styles.css) with a modern dark-themed glassmorphic UI:

    ```css
    :root {
        --bg-color: #0f0f15;
        --card-bg: rgba(25, 25, 35, 0.65);
        --accent-color: #00bcd4;
        --accent-hover: #00acc1;
        --text-primary: #f5f5f7;
        --text-secondary: #9e9eaf;
        --border-color: rgba(255, 255, 255, 0.08);
    }

    body {
        margin: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background-color: var(--bg-color);
        color: var(--text-primary);
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
    }

    .app-container {
        width: 100%;
        max-width: 1200px;
        padding: 20px;
        box-sizing: border-box;
    }

    header {
        text-align: center;
        margin-bottom: 30px;
    }

    header h1 {
        margin: 0;
        font-size: 2.2rem;
        background: linear-gradient(45deg, #00bcd4, #80deea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    header p {
        margin: 5px 0 0;
        color: var(--text-secondary);
    }

    .main-layout {
        display: grid;
        grid-template-columns: 2fr 1fr;
        gap: 20px;
    }

    @media (max-width: 768px) {
        .main-layout {
            grid-template-columns: 1fr;
        }
    }

    .video-section {
        display: flex;
        flex-direction: column;
        gap: 20px;
    }

    .video-wrapper {
        position: relative;
        background-color: #000;
        border-radius: 12px;
        overflow: hidden;
        aspect-ratio: 4/3;
        border: 1px solid var(--border-color);
    }

    video {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }

    canvas {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
    }

    .overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(15, 15, 21, 0.85);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 10;
        transition: opacity 0.3s ease;
    }

    .overlay.hidden {
        opacity: 0;
        pointer-events: none;
    }

    .spinner {
        width: 40px;
        height: 40px;
        border: 4px solid rgba(255, 255, 255, 0.1);
        border-top-color: var(--accent-color);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    .controls-panel {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        display: flex;
        flex-direction: column;
        gap: 15px;
        backdrop-filter: blur(10px);
    }

    .control-group {
        display: flex;
        flex-direction: column;
        gap: 5px;
    }

    .control-group label {
        font-size: 0.9rem;
        color: var(--text-secondary);
    }

    .control-group select, .control-group input[type="range"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        color: var(--text-primary);
        padding: 8px;
        outline: none;
    }

    .button-group {
        display: flex;
        gap: 10px;
        margin-top: 10px;
    }

    .btn {
        flex: 1;
        border: none;
        border-radius: 6px;
        padding: 10px 15px;
        font-weight: bold;
        cursor: pointer;
        transition: background-color 0.2s ease;
    }

    .btn.primary {
        background-color: var(--accent-color);
        color: #000;
    }

    .btn.primary:hover {
        background-color: var(--accent-hover);
    }

    .btn.secondary {
        background-color: rgba(255, 255, 255, 0.1);
        color: var(--text-primary);
    }

    .btn.secondary:hover {
        background-color: rgba(255, 255, 255, 0.15);
    }

    .btn:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .stats-section {
        display: flex;
        flex-direction: column;
        gap: 20px;
    }

    .card {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }

    .card h3 {
        margin-top: 0;
        margin-bottom: 15px;
        font-size: 1.1rem;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 10px;
    }

    .stat-row, .class-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    }

    .stat-row:last-child, .class-row:last-child {
        border-bottom: none;
    }

    .val {
        font-weight: bold;
        color: var(--accent-color);
    }
    ```

---

### Task 2: Implement Inference Controller Script

**Files:**
*   Create: `docs/app.js`

**Interfaces:**
*   Consumes: ONNX model binary files.
*   Produces: Image preprocessing logic (640x640 HWC -> 1x3x640x640 CHW), execution session, and NMS rendering.

*   [ ] **Step 1: Write app.js**
    Create [docs/app.js](file:///home/tgrwjya/Documents/Uni/Semester%206/AI_ML/CV/Project_Work/Fruits-detection/docs/app.js) with model configurations, camera setup, NMS loop, and rendering pipeline:

    ```javascript
    // Constants
    const CLASS_NAMES = ["Apple", "Banana", "Grape", "Orange", "Pineapple", "Watermelon"];
    const CLASS_COLORS = {
        "Apple":      "rgba(230, 50, 50, 1)",
        "Banana":     "rgba(250, 220, 30, 1)",
        "Grape":      "rgba(180, 60, 150, 1)",
        "Orange":     "rgba(255, 140, 40, 1)",
        "Pineapple":  "rgba(50, 200, 50, 1)",
        "Watermelon": "rgba(60, 180, 75, 1)"
    };

    // State Variables
    let session = null;
    let isRunning = false;
    let animationFrameId = null;

    // Elements
    const video = document.getElementById("webcam");
    const canvas = document.getElementById("detection-canvas");
    const ctx = canvas.getContext("2d");
    const loadingOverlay = document.getElementById("loading-overlay");
    const loadingText = document.getElementById("loading-text");
    const modelSelect = document.getElementById("model-select");
    const confSlider = document.getElementById("conf-threshold");
    const confVal = document.getElementById("conf-val");
    const iouSlider = document.getElementById("iou-threshold");
    const iouVal = document.getElementById("iou-val");
    const startBtn = document.getElementById("start-btn");
    const pauseBtn = document.getElementById("pause-btn");

    // Metric Displays
    const inferenceTimeDisplay = document.getElementById("inference-time");
    const fpsDisplay = document.getElementById("fps-counter");

    // Slider Listeners
    confSlider.addEventListener("input", (e) => confVal.textContent = parseFloat(e.target.value).toFixed(2));
    iouSlider.addEventListener("input", (e) => iouVal.textContent = parseFloat(e.target.value).toFixed(2));

    // Change Model Listener
    modelSelect.addEventListener("change", async () => {
        if (isRunning) {
            pauseDetection();
            await loadModel(modelSelect.value);
            startDetection();
        } else {
            await loadModel(modelSelect.value);
        }
    });

    // Button Listeners
    startBtn.addEventListener("click", async () => {
        if (!session) {
            await loadModel(modelSelect.value);
        }
        await startCamera();
        startDetection();
    });

    pauseBtn.addEventListener("click", () => {
        if (isRunning) {
            pauseDetection();
        } else {
            startDetection();
        }
    });

    // Load ONNX Model
    async function loadModel(modelName) {
        showLoading(`Loading ${modelName}...`);
        try {
            // Setup WebGL execution provider for GPU acceleration if available
            const options = { executionProviders: ['webgl', 'wasm'] };
            session = await ort.InferenceSession.create(modelName, options);
            hideLoading();
        } catch (err) {
            console.error("Failed to load model:", err);
            showLoading(`Error loading model: ${err.message}. Ensure the file exists in docs/`);
        }
    }

    // Setup Video Camera Stream
    async function startCamera() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480, facingMode: "environment" },
                    audio: false
                });
                video.srcObject = stream;
                return new Promise((resolve) => {
                    video.onloadedmetadata = () => {
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        resolve();
                    };
                });
            } catch (err) {
                alert(`Camera access blocked or not available: ${err.message}`);
                throw err;
            }
        } else {
            alert("getUserMedia is not supported by your browser");
        }
    }

    // Start/Pause Detection Loops
    function startDetection() {
        isRunning = true;
        pauseBtn.disabled = false;
        pauseBtn.textContent = "Pause";
        startBtn.disabled = true;
        renderLoop();
    }

    function pauseDetection() {
        isRunning = false;
        pauseBtn.textContent = "Resume";
        startBtn.disabled = false;
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
        }
    }

    // Performance FPS metrics
    let lastTime = 0;
    let frames = 0;
    let fps = 0;

    async function renderLoop() {
        if (!isRunning) return;

        const startTime = performance.now();
        await detectFrame();
        const duration = performance.now() - startTime;
        inferenceTimeDisplay.textContent = `${duration.toFixed(0)} ms`;

        // Calculate FPS
        frames++;
        const now = performance.now();
        if (now >= lastTime + 1000) {
            fps = (frames * 1000) / (now - lastTime);
            fpsDisplay.textContent = fps.toFixed(1);
            frames = 0;
            lastTime = now;
        }

        animationFrameId = requestAnimationFrame(renderLoop);
    }

    async function detectFrame() {
        if (!session) return;

        // Draw current video frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Preprocess frame to match input 1x3x640x640 CHW
        const inputTensor = preprocessCanvas(canvas);

        try {
            // Run inference
            const feeds = {};
            feeds[session.inputNames[0]] = inputTensor;
            const output = await session.run(feeds);
            const outputTensor = output[session.outputNames[0]];

            // Parse predictions
            const boxes = postprocess(outputTensor.data, parseFloat(confSlider.value), parseFloat(iouSlider.value));

            // Render boxes
            drawDetections(boxes);
        } catch (err) {
            console.error("Inference failed:", err);
        }
    }

    // Preprocess Canvas: Resize HWC -> CHW [1, 3, 640, 640]
    function preprocessCanvas(canvas) {
        // Create offscreen canvas for resizing
        const offscreen = document.createElement("canvas");
        offscreen.width = 640;
        offscreen.height = 640;
        const oCtx = offscreen.getContext("2d");
        oCtx.drawImage(canvas, 0, 0, 640, 640);

        const imgData = oCtx.getImageData(0, 0, 640, 640);
        const data = imgData.data;

        const r = new Float32Array(640 * 640);
        const g = new Float32Array(640 * 640);
        const b = new Float32Array(640 * 640);

        for (let i = 0; i < data.length; i += 4) {
            const pixelIdx = i / 4;
            r[pixelIdx] = data[i] / 255.0;
            g[pixelIdx] = data[i + 1] / 255.0;
            b[pixelIdx] = data[i + 2] / 255.0;
        }

        const tensorData = new Float32Array(3 * 640 * 640);
        tensorData.set(r, 0);
        tensorData.set(g, 640 * 640);
        tensorData.set(b, 2 * 640 * 640);

        return new ort.Tensor("float32", tensorData, [1, 3, 640, 640]);
    }

    // Postprocess: Parse predictions and apply NMS
    // Input flat array size: 10 * 8400 = 84000 values
    function postprocess(data, confThreshold, iouThreshold) {
        const candidates = [];
        const numClasses = CLASS_NAMES.length;
        const numAnchors = 8400;

        for (let col = 0; col < numAnchors; col++) {
            // Find class index with max score
            let maxClassIdx = -1;
            let maxScore = -1;

            for (let cls = 0; cls < numClasses; cls++) {
                const score = data[(cls + 4) * numAnchors + col];
                if (score > maxScore) {
                    maxScore = score;
                    maxClassIdx = cls;
                }
            }

            if (maxScore >= confThreshold) {
                // Read box centers
                const xc = data[0 * numAnchors + col];
                const yc = data[1 * numAnchors + col];
                const w = data[2 * numAnchors + col];
                const h = data[3 * numAnchors + col];

                // Convert to corners relative to 640x640 input
                const x1 = xc - w / 2;
                const y1 = yc - h / 2;
                const x2 = xc + w / 2;
                const y2 = yc + h / 2;

                candidates.push({
                    box: [x1, y1, x2, y2],
                    classIdx: maxClassIdx,
                    score: maxScore
                });
            }
        }

        // Sort candidates by score descending
        candidates.sort((a, b) => b.score - a.score);

        // Apply NMS
        const selected = [];
        for (const cand of candidates) {
            let keep = true;
            for (const sel of selected) {
                const iou = calculateIoU(cand.box, sel.box);
                if (iou >= iouThreshold) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                selected.push(cand);
            }
        }

        return selected;
    }

    // Intersection over Union (IoU) helper
    function calculateIoU(boxA, boxB) {
        const xA = Math.max(boxA[0], boxB[0]);
        const yA = Math.max(boxA[1], boxB[1]);
        const xB = Math.min(boxA[2], boxB[2]);
        const yB = Math.min(boxA[3], boxB[3]);

        const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
        if (interArea === 0) return 0;

        const boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[0]);
        const boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[0]);

        return interArea / (boxAArea + boxBArea - interArea);
    }

    // Render Bounding Boxes
    function drawDetections(boxes) {
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const counts = { "Apple": 0, "Banana": 0, "Grape": 0, "Orange": 0, "Pineapple": 0, "Watermelon": 0 };

        // Scale factors
        const scaleX = canvas.width / 640;
        const scaleY = canvas.height / 640;

        for (const item of boxes) {
            const [x1, y1, x2, y2] = item.box;
            const rx1 = x1 * scaleX;
            const ry1 = y1 * scaleY;
            const rx2 = x2 * scaleX;
            const ry2 = y2 * scaleY;
            const rw = rx2 - rx1;
            const rh = ry2 - ry1;

            const className = CLASS_NAMES[item.classIdx];
            const color = CLASS_COLORS[className];
            counts[className]++;

            // Draw Box
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(rx1, ry1, rw, rh);

            // Draw Corner Elements (Fancy style)
            const len = Math.min(rw, rh) * 0.2;
            ctx.fillStyle = color;
            ctx.fillRect(rx1 - 2, ry1 - 2, len, 4);
            ctx.fillRect(rx1 - 2, ry1 - 2, 4, len);
            ctx.fillRect(rx2 - len + 2, ry1 - 2, len, 4);
            ctx.fillRect(rx2 - 2, ry1 - 2, 4, len);
            ctx.fillRect(rx1 - 2, ry2 - 2, len, 4);
            ctx.fillRect(rx1 - 2, ry2 - len + 2, 4, len);
            ctx.fillRect(rx2 - len + 2, ry2 - 2, len, 4);
            ctx.fillRect(rx2 - 2, ry2 - len + 2, 4, len);

            // Draw Label background
            ctx.font = "bold 14px sans-serif";
            const text = `${className} ${(item.score * 100).toFixed(0)}%`;
            const textWidth = ctx.measureText(text).width;
            ctx.fillRect(rx1 - 2, ry1 - 22, textWidth + 10, 20);

            // Draw Text
            ctx.fillStyle = "#000";
            ctx.fillText(text, rx1 + 3, ry1 - 7);
        }

        // Update dashboard count numbers
        for (const cls of CLASS_NAMES) {
            document.getElementById(`count-${cls}`).textContent = counts[cls];
        }
    }

    // Helper loading overlays
    function showLoading(text) {
        loadingText.textContent = text;
        loadingOverlay.classList.remove("hidden");
    }

    function hideLoading() {
        loadingOverlay.classList.add("hidden");
    }

    // Initial default check
    window.addEventListener("DOMContentLoaded", () => {
        hideLoading();
    });
    ```

---

### Task 3: Setup GitHub Actions Deployment Workflow

**Files:**
*   Create: `.github/workflows/deploy.yml`

**Interfaces:**
*   Consumes: Push triggers to standard workspace branch `main`.
*   Produces: GitHub Pages deployment artifact routing `/docs` directory.

*   [ ] **Step 1: Create deploy.yml**
    Create [ .github/workflows/deploy.yml](file:///home/tgrwjya/Documents/Uni/Semester%206/AI_ML/CV/Project_Work/Fruits-detection/.github/workflows/deploy.yml):

    ```yaml
    name: Deploy to GitHub Pages

    on:
      push:
        branches: [main]
      workflow_dispatch:

    permissions:
      contents: read
      pages: write
      id-token: write

    concurrency:
      group: "pages"
      cancel-in-progress: false

    jobs:
      deploy:
        environment:
          name: github-pages
          url: ${{ steps.deployment.outputs.page_url }}
        runs-on: ubuntu-latest
        steps:
          - name: Checkout
            uses: actions/checkout@v4
          - name: Setup Pages
            uses: actions/configure-pages@v4
          - name: Upload artifact
            uses: actions/upload-pages-artifact@v3
            with:
              path: './docs'
          - name: Deploy to GitHub Pages
            id: deployment
            uses: actions/deploy-pages@v4
    ```

*   [ ] **Step 2: Commit Task 3 Workflow file**
    Notify the user before running git operations:
    ```bash
    rtk git add .github/workflows/deploy.yml
    rtk git commit -m "ci: add GitHub Actions workflow to auto deploy docs folder to Pages"
    ```

---

### Task 4: Local Quality Verification and Lighthouse Auditing

**Files:**
*   Test: Local environment (`http://localhost:8080/index.html`)

**Interfaces:**
*   Consumes: Static pages from the `docs/` folder.
*   Produces: Visual screenshots and a Lighthouse audit report.

*   [ ] **Step 1: Start a temporary local HTTP server**
    Run: `rtk python3 -m http.server 8080 --directory docs`
    *(Keep this running in the background as a task)*

*   [ ] **Step 2: Load the page in Chrome DevTools**
    Create a new browser page pointing to `http://localhost:8080/index.html` using the `new_page` tool.

*   [ ] **Step 3: Run Lighthouse audit**
    Execute a Lighthouse audit using `lighthouse_audit` to verify Performance, Accessibility (a11y), Best Practices, and SEO.

*   [ ] **Step 4: Capture page screenshot**
    Capture a screenshot using `take_screenshot` to verify visual styling and layout correctness. Save to `docs/web_screenshot.png`.

*   [ ] **Step 5: Clean up local HTTP server**
    Stop the background task of the Python HTTP server using `manage_task` with action `kill`.

