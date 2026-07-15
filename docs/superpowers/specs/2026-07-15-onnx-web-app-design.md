# Spec: YOLO ONNX Web Application & CI/CD Deployment

## Objective
Build a real-time web application to run YOLO fruit detection directly in the browser via WebAssembly (ONNX Runtime Web) using the device camera. Automate deployment of this static app to GitHub Pages using GitHub Actions.

## Tech Stack
*   HTML5 (webcam access & canvas drawing)
*   CSS3 (glassmorphism UI & custom styling)
*   Vanilla JavaScript (image preprocessing, model execution, and NMS postprocessing)
*   ONNX Runtime Web (`onnxruntime-web`) via CDN
*   GitHub Actions (for automated CI/CD page deployment)

## Architecture

```mermaid
graph TD
    A[Webcam Feed] --> B[HTML5 Video Element]
    B --> C[Canvas Image Extraction]
    C --> D[Image Preprocessing: Resize to 640x640, Normalize, Transpose CHW]
    D --> E[ONNX Runtime Web Session]
    E --> F[Inference: Output Tensor [1, 10, 8400]]
    F --> G[JavaScript Postprocessing: Confidence Filter & NMS]
    G --> H[Render Bounding Boxes and Dashboard Overlay]
```

### 1. Preprocessing Specs
*   **Resize:** Scale camera input frames to 640x640 pixels.
*   **Normalization:** Convert pixel values from [0-255] to [0.0-1.0] by dividing by 255.0.
*   **Transpose:** Convert the image layout from HWC (Height, Width, Channel) to CHW (Channel, Height, Width) to match the model's input dimension expectations of `[1, 3, 640, 640]`.

### 2. Postprocessing Specs
*   **Output Shape:** `[1, 10, 8400]`.
    *   Rows 0-3: Bounding box center coordinates and dimensions (`[xc, yc, w, h]`).
    *   Rows 4-9: Class scores for `[Apple, Banana, Grape, Orange, Pineapple, Watermelon]`.
*   **Confidence Filtering:** Ignore boxes where the maximum class probability is below the user-defined threshold (default: 0.40).
*   **Non-Maximum Suppression (NMS):** Perform NMS using intersection-over-union (IoU) to filter overlapping boxes (default threshold: 0.45).

### 3. CI/CD Pipeline (`.github/workflows/deploy.yml`)
*   Triggered on pushes to `main`.
*   Quality Gates: Runs HTML/JS linting or validation checks, and deploys the static files inside the `docs/` directory to GitHub Pages.

## Verification Plan
1.  **Local Run:** Load `docs/index.html` via a local HTTP server and verify the webcam requests access and feeds correctly.
2.  **Inference Check:** Load the ONNX model in browser and verify there are no WebGL/WebAssembly load crashes.
3.  **CI Run:** Verify GitHub Actions run runs cleanly without syntax errors in the YAML definition.

## Safety & Style Guidelines
*   **No Emojis:** Do NOT use emojis in code, comments, or documentation.
*   **RTK CLI Prefix:** Prefix all CLI commands with `rtk`.
