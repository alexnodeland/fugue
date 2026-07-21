/* fugue-wasm loader — makes the REAL fugue crate available to the doc
 * widgets and the playground.
 *
 * Publishes `FV.wasmReady`, a promise resolving to the wasm-bindgen module
 * namespace (WasmMh, WasmHmc, WasmParticleFilter, wasm_smc_run,
 * log_joint_grid, check_model, ...) once `<site>/pkg/fugue_wasm.js` has
 * loaded and initialized — or to `null` when the package is absent (local
 * builds without wasm-pack) or the browser cannot load it. Widgets await
 * this promise at init time and fall back to their mirrored-JS math cores
 * on `null`, so the docs degrade gracefully.
 *
 * Must be listed in book.toml after fugue-viz.js and before viz/*.js.
 */
(function () {
  'use strict';
  if (!window.FugueViz) return;
  var FV = window.FugueViz;

  var script =
    document.currentScript ||
    document.querySelector('script[src*="fugue-wasm-loader"]');
  var root = script ? script.src.replace(/fugue-wasm-loader\.js.*$/, '') : './';

  // Dynamic import via Function so browsers without import() support fail
  // at call time (caught below), not while parsing this whole file.
  var dynImport;
  try {
    dynImport = new Function('u', 'return import(u)');
  } catch (e) {
    FV.wasm = null;
    FV.wasmReady = Promise.resolve(null);
    return;
  }

  FV.wasm = null;
  FV.wasmReady = dynImport(root + 'pkg/fugue_wasm.js')
    .then(function (mod) {
      return mod.default(root + 'pkg/fugue_wasm_bg.wasm').then(function () {
        FV.wasm = mod;
        try {
          console.log(
            '[fugue-viz] fugue-wasm ' +
              mod.fugue_version() +
              ' loaded — widgets run the real crate'
          );
        } catch (e) { /* readout only */ }
        return mod;
      });
    })
    .catch(function (e) {
      try {
        console.log(
          '[fugue-viz] fugue-wasm unavailable (' +
            (e && e.message ? e.message : e) +
            ') — widgets use the mirrored JS math'
        );
      } catch (e2) { /* readout only */ }
      return null;
    });
})();
