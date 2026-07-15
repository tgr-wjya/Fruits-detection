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

// Pause loop
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

    const interWidth = Math.max(0, xB - xA);
    const interHeight = Math.max(0, yB - yA);
    const interArea = interWidth * interHeight;
    if (interArea === 0) return 0;

    const boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
    const boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);

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
