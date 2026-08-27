/* Tests for the on-device detector's coordinate maths.
 *
 * These are the calculations that fail silently. A wrong letterbox inverse
 * still produces plausible-looking boxes on screen; they are simply in the
 * wrong place, and every distance, width and severity derived from them is
 * quietly wrong. Run with:  node tests/test_detector.mjs
 */
import assert from 'node:assert/strict';
import { letterboxParams, decodeOutput, nms, INPUT_SIZE }
  from '../app/static/detector.js';

let passed = 0;
const test = (name, fn) => {
  try { fn(); passed++; console.log(`  ok   ${name}`); }
  catch (e) { console.error(`  FAIL ${name}\n       ${e.message}`); process.exitCode = 1; }
};

/* Build a fake YOLOv8 output containing one box, in letterboxed pixels. */
function fakeOutput({ cx, cy, w, h, score, anchors = 4, numClasses = 1 }) {
  const channels = 4 + numClasses;
  const data = new Float32Array(channels * anchors);
  data[0] = cx; data[anchors] = cy;
  data[2 * anchors] = w; data[3 * anchors] = h;
  data[4 * anchors] = score;
  return { data, dims: [1, channels, anchors] };
}

console.log('on-device detector');

test('letterbox centres a landscape frame with no distortion', () => {
  const { scale, dx, dy } = letterboxParams(1280, 720);
  assert.equal(scale, INPUT_SIZE / 1280);
  assert.equal(dx, 0);
  assert.equal(dy, Math.floor((INPUT_SIZE - Math.round(720 * scale)) / 2));
});

test('a box round-trips back to its original frame coordinates', () => {
  const sw = 1280, sh = 720;
  const { scale, dx, dy } = letterboxParams(sw, sh);
  // A defect occupying a known rectangle of the real frame...
  const want = [500, 400, 620, 460];
  const cx = ((want[0] + want[2]) / 2) * scale + dx;
  const cy = ((want[1] + want[3]) / 2) * scale + dy;
  const w = (want[2] - want[0]) * scale, h = (want[3] - want[1]) * scale;

  const { data, dims } = fakeOutput({ cx, cy, w, h, score: 0.9 });
  const [box] = decodeOutput(data, dims, { scale, dx, dy, sw, sh }, 0.45, 0.5);
  box.bbox.forEach((v, i) => assert.ok(Math.abs(v - want[i]) < 1e-6,
    `edge ${i}: got ${v}, want ${want[i]}`));
});

test('padding is removed before scaling, not after', () => {
  // A box at the very top of the padded image must map to the top of the
  // frame. Dividing before subtracting would push it off by dy/scale pixels.
  const sw = 1280, sh = 720;
  const { scale, dx, dy } = letterboxParams(sw, sh);
  const { data, dims } = fakeOutput({ cx: INPUT_SIZE / 2, cy: dy + 10 * scale,
                                      w: 20 * scale, h: 20 * scale, score: 0.9 });
  const [box] = decodeOutput(data, dims, { scale, dx, dy, sw, sh }, 0.45, 0.5);
  assert.ok(Math.abs(box.bbox[1] - 0) < 1e-6, `top edge was ${box.bbox[1]}`);
});

test('detections below the confidence threshold are dropped', () => {
  const { data, dims } = fakeOutput({ cx: 320, cy: 320, w: 40, h: 40, score: 0.2 });
  const out = decodeOutput(data, dims, { scale: 1, dx: 0, dy: 0, sw: 640, sh: 640 },
                           0.45, 0.5);
  assert.equal(out.length, 0);
});

test('boxes are clamped to the frame', () => {
  const { data, dims } = fakeOutput({ cx: 5, cy: 5, w: 60, h: 60, score: 0.9 });
  const [box] = decodeOutput(data, dims, { scale: 1, dx: 0, dy: 0, sw: 640, sh: 480 },
                             0.45, 0.5);
  assert.ok(box.bbox[0] >= 0 && box.bbox[1] >= 0);
});

test('overlapping boxes collapse to the most confident one', () => {
  const kept = nms([
    { confidence: 0.9, bbox: [100, 100, 200, 200] },
    { confidence: 0.7, bbox: [105, 105, 205, 205] },   // same pothole
    { confidence: 0.8, bbox: [400, 400, 500, 500] },   // a different one
  ], 0.5);
  assert.equal(kept.length, 2);
  assert.equal(kept[0].confidence, 0.9);
});

test('a multi-class model takes the strongest class', () => {
  const anchors = 2, numClasses = 3, channels = 4 + numClasses;
  const data = new Float32Array(channels * anchors);
  data[0] = 320; data[anchors] = 320;
  data[2 * anchors] = 40; data[3 * anchors] = 40;
  data[4 * anchors] = 0.10;      // pothole
  data[5 * anchors] = 0.82;      // crack
  data[6 * anchors] = 0.30;      // manhole
  const [box] = decodeOutput(data, [1, channels, anchors],
                             { scale: 1, dx: 0, dy: 0, sw: 640, sh: 640 }, 0.45, 0.5);
  assert.ok(Math.abs(box.confidence - 0.82) < 1e-6);
});

console.log(`\n${passed} passed`);
