# Vendored ONNX Runtime Web

Files copied verbatim from the `onnxruntime-web` npm package, MIT licensed,
(c) Microsoft Corporation. See https://github.com/microsoft/onnxruntime.

Only the WebAssembly backend is vendored:

| File | Purpose |
|---|---|
| `ort.wasm.min.js` | Runtime loader, WASM backend only |
| `ort-wasm-simd-threaded.wasm` | The runtime itself (SIMD, threaded build) |
| `ort-wasm-simd-threaded.mjs` | Its JavaScript glue |

They are served from this repository rather than a CDN for the same reason
Leaflet is: the phone must be able to run a survey with no internet
connection, and a car is exactly where you lose one.
