/* On-device YOLO inference for the phone client.
 *
 * Why this exists
 * ---------------
 * Streaming frames to a laptop means a laptop has to be in the car, on the
 * same network, powered, for the whole drive. Running the model on the phone
 * removes all of that: no laptop, no hotspot, no upload of road imagery at
 * all. It is also the only version of this that a person would actually use
 * more than once.
 *
 * The runtime is ONNX Runtime Web on the WebAssembly backend, vendored so a
 * survey works with no internet. The model is the same yolov8n exported from
 * the training notebook, so the phone and the server run identical weights.
 *
 * Everything below is the unglamorous half of "run YOLO in a browser": get
 * the pixels into the tensor layout the network expects, and turn its raw
 * output back into boxes in the original frame's coordinates.
 */

export const INPUT_SIZE = 640;

const RUNTIME_URL = '/static/vendor/ort/ort.wasm.min.js';
let runtimeLoading = null;

/* Pull in ONNX Runtime Web on first use.
 *
 * It ships as a classic script that assigns a global, not an ES module, so it
 * is injected rather than imported. Deliberately lazy: a phone surveying in
 * server mode never runs a model and should not pay 11 MB to find that out,
 * and the service worker caches it on the first on-device run so later drives
 * need no network at all. */
function ensureRuntime() {
  if (globalThis.ort) return Promise.resolve(globalThis.ort);
  if (runtimeLoading) return runtimeLoading;
  runtimeLoading = new Promise((resolve, reject) => {
    const tag = document.createElement('script');
    tag.src = RUNTIME_URL;
    tag.onload = () => globalThis.ort
      ? resolve(globalThis.ort)
      : reject(new Error('runtime loaded but exposed no ort global'));
    tag.onerror = () => reject(new Error(`could not load ${RUNTIME_URL}`));
    document.head.appendChild(tag);
  });
  return runtimeLoading;
}

/* Letterbox to a square: resize preserving aspect ratio, pad the remainder.
 * Squashing the frame instead would distort every box, and the whole
 * downstream geometry - distance, width, severity - depends on box edges
 * being where the road really is.
 *
 * Exported separately from the drawing so the mapping can be tested without
 * a browser. Getting this wrong is silent: boxes still appear, they are just
 * in the wrong place, and every measurement downstream is quietly wrong. */
export function letterboxParams(sw, sh) {
  const scale = Math.min(INPUT_SIZE / sw, INPUT_SIZE / sh);
  const w = Math.round(sw * scale), h = Math.round(sh * scale);
  return { scale, dx: Math.floor((INPUT_SIZE - w) / 2),
           dy: Math.floor((INPUT_SIZE - h) / 2) };
}

function letterbox(source, sw, sh, ctx) {
  const { scale, dx, dy } = letterboxParams(sw, sh);
  ctx.fillStyle = '#727272';                       // neutral grey padding
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  ctx.drawImage(source, 0, 0, sw, sh, dx, dy, Math.round(sw * scale),
                Math.round(sh * scale));
  return { scale, dx, dy };
}

/* RGBA bytes -> planar RGB float32 in [0,1], the NCHW layout YOLO expects. */
function toTensorData(rgba, out) {
  const px = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0, p = 0; p < px; i += 4, p++) {
    out[p] = rgba[i] / 255;
    out[px + p] = rgba[i + 1] / 255;
    out[2 * px + p] = rgba[i + 2] / 255;
  }
  return out;
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (inter <= 0) return 0;
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter);
}

export function nms(boxes, threshold) {
  const kept = [];
  boxes.sort((p, q) => q.confidence - p.confidence);
  for (const box of boxes) {
    if (!kept.some(k => iou(k.bbox, box.bbox) > threshold)) kept.push(box);
  }
  return kept;
}

/* Turn YOLOv8's raw output into boxes in the original frame's coordinates.
 *
 * The network emits [1, 4+numClasses, anchors]: centre-x, centre-y, width and
 * height in letterboxed pixels, then one score per class. Undoing the
 * letterbox means subtracting the padding before dividing by the scale, in
 * that order - the reverse quietly shifts every box by a few pixels, which at
 * 25 m is metres of localisation error. */
export function decodeOutput(data, dims, geom, confThreshold, iouThreshold) {
  const [, channels, anchors] = dims;
  const { scale, dx, dy, sw, sh } = geom;
  const numClasses = channels - 4;

  const candidates = [];
  for (let a = 0; a < anchors; a++) {
    // Best class score at this anchor. A single-class pothole model has one,
    // but reading it generically means a multi-class defect model
    // (pothole / crack / manhole cover) drops straight in.
    let best = 0;
    for (let c = 0; c < numClasses; c++) {
      const score = data[(4 + c) * anchors + a];
      if (score > best) best = score;
    }
    if (best < confThreshold) continue;

    const cx = data[a], cy = data[anchors + a];
    const w = data[2 * anchors + a], h = data[3 * anchors + a];
    candidates.push({
      confidence: best,
      bbox: [
        (cx - w / 2 - dx) / scale, (cy - h / 2 - dy) / scale,
        (cx + w / 2 - dx) / scale, (cy + h / 2 - dy) / scale,
      ],
    });
  }

  return nms(candidates, iouThreshold).map(b => ({
    confidence: b.confidence,
    bbox: [
      Math.max(0, b.bbox[0]), Math.max(0, b.bbox[1]),
      Math.min(sw, b.bbox[2]), Math.min(sh, b.bbox[3]),
    ],
  }));
}


export class OnDeviceDetector {
  constructor({ modelUrl, confThreshold = 0.45, iouThreshold = 0.5 }) {
    this.modelUrl = modelUrl;
    this.confThreshold = confThreshold;
    this.iouThreshold = iouThreshold;
    this.session = null;
    this.ort = null;
    this.threads = 1;

    this.canvas = document.createElement('canvas');
    this.canvas.width = this.canvas.height = INPUT_SIZE;
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    this.buffer = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  }

  async load(onProgress) {
    const ort = await ensureRuntime();
    ort.env.wasm.wasmPaths = '/static/vendor/ort/';
    // Threads need SharedArrayBuffer, which needs a cross-origin isolated
    // page. The server sets COOP/COEP for exactly this; if something strips
    // them we still run, just on one core.
    this.threads = globalThis.crossOriginIsolated
      ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    ort.env.wasm.numThreads = this.threads;

    // Fetch the weights ourselves so download progress can be shown and so
    // the service worker can serve them from cache on a later, offline drive.
    const res = await fetch(this.modelUrl);
    if (!res.ok) throw new Error(`model download failed (${res.status})`);
    const total = Number(res.headers.get('content-length')) || 0;
    const chunks = [];
    let received = 0;
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      if (onProgress) onProgress(received, total);
    }
    const bytes = new Uint8Array(received);
    let at = 0;
    for (const c of chunks) { bytes.set(c, at); at += c.length; }

    this.ort = ort;
    this.session = await ort.InferenceSession.create(bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    this.inputName = this.session.inputNames[0];
    return { bytes: received, threads: this.threads };
  }

  get ready() { return this.session !== null; }

  /* Detect in a video frame. Returns boxes in that frame's pixel
   * coordinates, so the server's geometry needs no change at all. */
  async detect(video) {
    const sw = video.videoWidth, sh = video.videoHeight;
    const { scale, dx, dy } = letterbox(video, sw, sh, this.ctx);
    const rgba = this.ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
    const tensor = new this.ort.Tensor(
      'float32', toTensorData(rgba, this.buffer), [1, 3, INPUT_SIZE, INPUT_SIZE]);

    const started = performance.now();
    const output = await this.session.run({ [this.inputName]: tensor });
    const inferenceMs = performance.now() - started;

    const raw = output[this.session.outputNames[0]];
    const boxes = decodeOutput(raw.data, raw.dims, { scale, dx, dy, sw, sh },
                               this.confThreshold, this.iouThreshold);
    return { boxes, inferenceMs, frameW: sw, frameH: sh };
  }
}
