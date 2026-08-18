(function (root) {
  "use strict";

  const MIN_ZOOM = 0.05;
  const MAX_ZOOM = 32;
  const FIXED_ZOOMS = [2, 4, 8, 16];

  function frozenPoint(sample) {
    return Object.freeze({ row: Number(sample.row), column: Number(sample.column) });
  }

  function createState(sample, fullWidth, fullHeight, layer = "natural_rgb") {
    const frozen = frozenPoint(sample);
    return {
      frozen,
      inspection: { row: frozen.row, column: frozen.column },
      center: { x: fullWidth / 2, y: fullHeight / 2 },
      fullWidth,
      fullHeight,
      zoom: 0,
      layer,
    };
  }

  function inspect(state, row, column) {
    return {
      ...state,
      inspection: {
        row: Math.max(0, Math.min(state.fullHeight - 1, Math.floor(row))),
        column: Math.max(0, Math.min(state.fullWidth - 1, Math.floor(column))),
      },
    };
  }

  function pan(state, deltaScreenX, deltaScreenY, pixelsPerFullPixel) {
    if (!(pixelsPerFullPixel > 0)) return state;
    return {
      ...state,
      center: {
        x: state.center.x - deltaScreenX / pixelsPerFullPixel,
        y: state.center.y - deltaScreenY / pixelsPerFullPixel,
      },
    };
  }

  function setZoom(state, zoom, anchor) {
    const next = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, Number(zoom)));
    if (!anchor || !(anchor.oldZoom > 0)) return { ...state, zoom: next };
    return {
      ...state,
      zoom: next,
      center: {
        x: anchor.fullX - anchor.offsetX / next,
        y: anchor.fullY - anchor.offsetY / next,
      },
    };
  }

  function reset(state) {
    return {
      ...state,
      inspection: { row: state.frozen.row, column: state.frozen.column },
      center: { x: state.fullWidth / 2, y: state.fullHeight / 2 },
      zoom: 0,
    };
  }

  function switchLayer(state, layer) {
    return { ...state, layer };
  }

  const Model = { createState, inspect, pan, setZoom, reset, switchLayer, FIXED_ZOOMS };
  if (typeof module !== "undefined" && module.exports) module.exports = Model;
  root.Field2ViewerModel = Model;
  if (typeof document === "undefined") return;

  const $ = (id) => document.getElementById(id);
  const viewer = $("viewer");
  const viewerCtx = viewer.getContext("2d");
  const frozenMagnifier = $("frozenMagnifier");
  const frozenCtx = frozenMagnifier.getContext("2d");
  const inspectionMagnifier = $("inspectionMagnifier");
  const inspectionCtx = inspectionMagnifier.getContext("2d");
  let samples = [];
  let visible = [];
  let manifest = {};
  let schema = {};
  let annotations = {};
  let sampleIndex = 0;
  let image = new Image();
  let viewerState = null;
  let dirty = false;
  let drag = null;
  let selectedLabel = "";
  let selectedConfidence = "";

  function currentSample() { return visible[sampleIndex]; }
  function currentRecord() { return annotations.annotations[currentSample().sample_id]; }
  function setStatus(message, bad = dirty) {
    $("status").textContent = message;
    $("status").className = bad ? "dirty" : "";
  }
  function setDirty() {
    dirty = true;
    setStatus("UNSAVED changes — use Save all.", true);
  }
  function layerStep() {
    const item = manifest[currentSample().cube_id];
    return viewerState.layer === "natural_rgb" ? Number(item.natural_rgb_preview_step) : Number(item.preview_step);
  }
  function fullDimensions() {
    const item = manifest[currentSample().cube_id];
    return { width: Number(item.natural_rgb_width), height: Number(item.natural_rgb_height) };
  }
  function fitZoom(canvas, step) {
    return Math.min(canvas.width / (image.naturalWidth * step), canvas.height / (image.naturalHeight * step));
  }
  function effectiveZoom(canvas = viewer) {
    return viewerState.zoom > 0 ? viewerState.zoom : fitZoom(canvas, layerStep());
  }
  function clampStateCenter(state, zoom) {
    const halfW = viewer.width / (2 * zoom);
    const halfH = viewer.height / (2 * zoom);
    const x = halfW >= state.fullWidth / 2 ? state.fullWidth / 2 : Math.max(halfW, Math.min(state.fullWidth - halfW, state.center.x));
    const y = halfH >= state.fullHeight / 2 ? state.fullHeight / 2 : Math.max(halfH, Math.min(state.fullHeight - halfH, state.center.y));
    return { ...state, center: { x, y } };
  }
  function transform(canvas, center, zoom) {
    return {
      x: (fullX) => canvas.width / 2 + (fullX - center.x) * zoom,
      y: (fullY) => canvas.height / 2 + (fullY - center.y) * zoom,
    };
  }
  function drawPixelGrid(ctx, canvas, center, zoom) {
    if (!$("grid").checked || !FIXED_ZOOMS.includes(viewerState.zoom) || viewerState.zoom < 8) return;
    const map = transform(canvas, center, zoom);
    const left = Math.max(0, Math.floor(center.x - canvas.width / (2 * zoom)));
    const right = Math.min(viewerState.fullWidth, Math.ceil(center.x + canvas.width / (2 * zoom)));
    const top = Math.max(0, Math.floor(center.y - canvas.height / (2 * zoom)));
    const bottom = Math.min(viewerState.fullHeight, Math.ceil(center.y + canvas.height / (2 * zoom)));
    ctx.strokeStyle = "rgba(255,255,255,.25)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let column = left; column <= right; column += 1) {
      const x = Math.round(map.x(column)) + 0.5;
      ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height);
    }
    for (let row = top; row <= bottom; row += 1) {
      const y = Math.round(map.y(row)) + 0.5;
      ctx.moveTo(0, y); ctx.lineTo(canvas.width, y);
    }
    ctx.stroke();
  }
  function drawFrozen(ctx, map, zoom) {
    const point = viewerState.frozen;
    const size = Math.max(4, zoom);
    const centerX = map.x(point.column + 0.5);
    const centerY = map.y(point.row + 0.5);
    ctx.save();
    ctx.strokeStyle = "#ffd166";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(centerX - size / 2 + 0.75, centerY - size / 2 + 0.75, size - 1.5, size - 1.5);
    ctx.restore();
  }
  function drawInspection(ctx, map, zoom) {
    const point = viewerState.inspection;
    const x = map.x(point.column + 0.5);
    const y = map.y(point.row + 0.5);
    const gap = Math.max(3, zoom * 0.65);
    const arm = Math.max(9, zoom * 1.5);
    ctx.save();
    ctx.strokeStyle = "#67e8f9";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x - arm, y); ctx.lineTo(x - gap, y);
    ctx.moveTo(x + gap, y); ctx.lineTo(x + arm, y);
    ctx.moveTo(x, y - arm); ctx.lineTo(x, y - gap);
    ctx.moveTo(x, y + gap); ctx.lineTo(x, y + arm);
    ctx.stroke();
    ctx.restore();
  }
  function drawImage(ctx, canvas, center, zoom) {
    const step = layerStep();
    const sourceScale = zoom * step;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = zoom < 2;
    const dx = canvas.width / 2 - (center.x / step) * sourceScale;
    const dy = canvas.height / 2 - (center.y / step) * sourceScale;
    ctx.drawImage(image, dx, dy, image.naturalWidth * sourceScale, image.naturalHeight * sourceScale);
    drawPixelGrid(ctx, canvas, center, zoom);
    return transform(canvas, center, zoom);
  }
  function drawViewer() {
    if (!viewerState || !image.complete || !image.naturalWidth) return;
    const zoom = effectiveZoom();
    viewerState = clampStateCenter(viewerState, zoom);
    const map = drawImage(viewerCtx, viewer, viewerState.center, zoom);
    drawFrozen(viewerCtx, map, zoom);
    drawInspection(viewerCtx, map, zoom);
  }
  function magnifierZoom() { return viewerState.zoom >= 2 ? viewerState.zoom : 8; }
  function drawMagnifier(ctx, canvas, point, marker) {
    const zoom = magnifierZoom();
    const center = { x: point.column + 0.5, y: point.row + 0.5 };
    const map = drawImage(ctx, canvas, center, zoom);
    if (marker === "frozen") drawFrozen(ctx, map, zoom);
    else drawInspection(ctx, map, zoom);
  }
  function renderCanvases() {
    drawViewer();
    drawMagnifier(frozenCtx, frozenMagnifier, viewerState.frozen, "frozen");
    if ($("showInspection").checked) drawMagnifier(inspectionCtx, inspectionMagnifier, viewerState.inspection, "inspection");
  }
  function chooseButtons(containerId, selected) {
    for (const button of $(containerId).querySelectorAll("button")) button.setAttribute("aria-pressed", String(button.dataset.value === selected));
  }
  function renderForm() {
    const sample = currentSample();
    const record = currentRecord();
    selectedLabel = record.selected_label;
    selectedConfidence = record.confidence;
    chooseButtons("labelButtons", selectedLabel);
    chooseButtons("confidenceButtons", selectedConfidence);
    $("reviewer").value = record.reviewer_identifier;
    $("note").value = record.investigator_note;
    $("review").textContent = record.reviewed && !record.requires_visual_rereview ? "Reviewed ✓" : "Mark reviewed";
    $("rereview").textContent = record.requires_visual_rereview ? "VISUAL RE-REVIEW REQUIRED: this preserved label predates the frozen natural RGB display." : "";
    $("contradiction").textContent = record.role_contradiction ? "ROLE CONTRADICTION RECORDED: investigator selected chickpea on a frozen absent-negative-control cube." : "";
    const item = manifest[sample.cube_id];
    const wavelengths = item.natural_rgb_wavelengths_nm;
    $("metadata").textContent = `sample ID: ${sample.sample_id}\ncube ID: ${sample.cube_id}\nfrozen row, column: ${sample.row}, ${sample.column}\ninspection row, column: ${viewerState.inspection.row}, ${viewerState.inspection.column}\nfrozen cube role: ${sample.cube_evaluation_role}\ncube notes: ${sample.cube_investigator_notes || "(none)"}\nRGB wavelengths: ${wavelengths.red} / ${wavelengths.green} / ${wavelengths.blue} nm\ndisplay version: ${record.annotation_display_version}\nrank stratum: ${sample.scalar_index_rank_stratum}\nspatial block: ${sample.spatial_group_id}\nframe: MAIN only`;
    const reviewed = Object.values(annotations.annotations).filter((item) => item.reviewed).length;
    const rereview = Object.values(annotations.annotations).filter((item) => item.requires_visual_rereview).length;
    $("counter").textContent = `${sampleIndex + 1} / ${visible.length} filtered · ${reviewed}/800 reviewed · ${rereview} re-review`;
    setStatus(`${sample.sample_id} · ${record.reviewed ? "REVIEWED" : "not reviewed"}${record.requires_visual_rereview ? " · RE-REVIEW REQUIRED" : ""} · ${record.selected_label || "unlabeled"}${dirty ? " · UNSAVED" : ""}`, dirty);
    for (const button of document.querySelectorAll(".zoom-choice")) button.classList.toggle("active", Number(button.dataset.zoom) === viewerState.zoom);
  }
  function render() { renderCanvases(); renderForm(); }
  function loadImage(resetSample) {
    const sample = currentSample();
    if (!sample) return;
    const dimensions = fullDimensions();
    if (resetSample || !viewerState) viewerState = createState(sample, dimensions.width, dimensions.height, $("layer").value);
    else viewerState = switchLayer(viewerState, $("layer").value);
    image = new Image();
    image.onload = render;
    image.onerror = () => setStatus("Prediction-free display layer failed to load.", true);
    image.src = `/layers/${sample.cube_id}/${viewerState.layer}.png?v=${Date.now()}`;
  }
  function filterSamples() {
    const cube = $("cubeFilter").value;
    const role = $("roleFilter").value;
    const review = $("reviewFilter").value;
    const currentId = currentSample() && currentSample().sample_id;
    visible = samples.filter((item) => (!cube || item.cube_id === cube) && (!role || item.cube_evaluation_role === role) && (!review || annotations.annotations[item.sample_id].requires_visual_rereview));
    const found = visible.findIndex((item) => item.sample_id === currentId);
    sampleIndex = found >= 0 ? found : 0;
    if (!visible.length) return setStatus("No MAIN samples match this filter.", true);
    loadImage(true);
  }
  function move(delta) {
    sampleIndex = Math.max(0, Math.min(visible.length - 1, sampleIndex + delta));
    loadImage(true);
  }
  function updateAnnotation() {
    const sample = currentSample();
    const record = currentRecord();
    record.selected_label = selectedLabel;
    record.confidence = selectedConfidence;
    record.reviewer_identifier = $("reviewer").value.trim();
    record.investigator_note = $("note").value;
    record.role_contradiction = selectedLabel === "chickpea" && sample.cube_evaluation_role === "chickpea_absent_negative_control";
    record.reviewed = false;
    record.review_timestamp = "";
    setDirty();
    renderForm();
  }
  function setChoice(field, value) {
    if (field === "selected_label") selectedLabel = selectedLabel === value ? "" : value;
    else selectedConfidence = selectedConfidence === value ? "" : value;
    updateAnnotation();
  }
  async function save() {
    const response = await fetch("/api/annotations", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(annotations) });
    if (!response.ok) return setStatus(await response.text(), true);
    const result = await response.json();
    annotations.revision = result.revision;
    dirty = false;
    setStatus(`SAVED: ${result.records} MAIN records · ${result.reviewed} reviewed. File: ${result.file}`);
  }
  function eventPosition(event, canvas) {
    const rect = canvas.getBoundingClientRect();
    return { x: (event.clientX - rect.left) * canvas.width / rect.width, y: (event.clientY - rect.top) * canvas.height / rect.height };
  }
  function screenToFull(position) {
    const zoom = effectiveZoom();
    return { x: viewerState.center.x + (position.x - viewer.width / 2) / zoom, y: viewerState.center.y + (position.y - viewer.height / 2) / zoom };
  }
  function bindViewerEvents() {
    viewer.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      const position = eventPosition(event, viewer);
      drag = { pointerId: event.pointerId, start: position, last: position, moved: false };
      try { viewer.setPointerCapture(event.pointerId); } catch (_) { /* Synthetic test events have no active pointer. */ }
    });
    viewer.addEventListener("pointermove", (event) => {
      if (!drag || drag.pointerId !== event.pointerId) return;
      const position = eventPosition(event, viewer);
      const dx = position.x - drag.last.x;
      const dy = position.y - drag.last.y;
      if (Math.hypot(position.x - drag.start.x, position.y - drag.start.y) > 3) drag.moved = true;
      if (drag.moved) {
        viewerState = pan(viewerState, dx, dy, effectiveZoom());
        viewer.classList.add("dragging");
        drag.last = position;
        renderCanvases();
      }
    });
    function finishPointer(event) {
      if (!drag || drag.pointerId !== event.pointerId) return;
      const moved = drag.moved;
      const position = eventPosition(event, viewer);
      drag = null;
      viewer.classList.remove("dragging");
      if (!moved) {
        const full = screenToFull(position);
        viewerState = inspect(viewerState, full.y, full.x);
      }
      render();
    }
    viewer.addEventListener("pointerup", finishPointer);
    viewer.addEventListener("pointercancel", (event) => { if (drag && drag.pointerId === event.pointerId) { drag = null; viewer.classList.remove("dragging"); } });
    viewer.addEventListener("wheel", (event) => {
      event.preventDefault();
      const position = eventPosition(event, viewer);
      const oldZoom = effectiveZoom();
      const full = screenToFull(position);
      const nextZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, oldZoom * Math.exp(-event.deltaY * 0.0015)));
      viewerState = setZoom(viewerState, nextZoom, { oldZoom, fullX: full.x, fullY: full.y, offsetX: position.x - viewer.width / 2, offsetY: position.y - viewer.height / 2 });
      render();
    }, { passive: false });
  }
  function buildChoices() {
    for (const value of schema.labels) {
      const button = document.createElement("button");
      button.type = "button"; button.className = "choice label-choice"; button.dataset.value = value; button.textContent = value.replaceAll("_", " "); button.setAttribute("aria-pressed", "false");
      button.addEventListener("click", () => setChoice("selected_label", value));
      $("labelButtons").appendChild(button);
    }
    for (const value of schema.confidence) {
      const button = document.createElement("button");
      button.type = "button"; button.className = "choice confidence-choice"; button.dataset.value = value; button.textContent = value; button.setAttribute("aria-pressed", "false");
      button.addEventListener("click", () => setChoice("confidence", value));
      $("confidenceButtons").appendChild(button);
    }
  }
  function bindControls() {
    $("prev").addEventListener("click", () => move(-1));
    $("next").addEventListener("click", () => move(1));
    for (const id of ["cubeFilter", "roleFilter", "reviewFilter"]) $(id).addEventListener("change", filterSamples);
    $("layer").addEventListener("change", () => loadImage(false));
    for (const button of document.querySelectorAll(".zoom-choice")) button.addEventListener("click", () => { viewerState = setZoom(viewerState, Number(button.dataset.zoom)); render(); });
    $("resetView").addEventListener("click", () => { viewerState = reset(viewerState); render(); });
    $("grid").addEventListener("change", renderCanvases);
    $("showInspection").addEventListener("change", () => { $("inspectionPanel").classList.toggle("hidden", !$("showInspection").checked); renderCanvases(); });
    $("reviewer").addEventListener("change", updateAnnotation);
    $("note").addEventListener("change", updateAnnotation);
    $("review").addEventListener("click", () => {
      updateAnnotation();
      const record = currentRecord();
      if (!selectedLabel || !selectedConfidence) return setStatus("Select a label and confidence before marking reviewed.", true);
      record.reviewed = true; record.review_timestamp = new Date().toISOString(); record.requires_visual_rereview = false; setDirty(); render();
    });
    $("clear").addEventListener("click", () => {
      if (!confirm("Clear this MAIN annotation?")) return;
      const sample = currentSample(); const old = currentRecord();
      annotations.annotations[sample.sample_id] = { sample_id: sample.sample_id, cube_id: sample.cube_id, frozen_cube_role: sample.cube_evaluation_role, selected_label: "", confidence: "", investigator_note: "", reviewed: false, review_timestamp: "", reviewer_identifier: "", role_contradiction: false, source_preview_checksums: JSON.parse(sample.source_preview_checksums), sampling_frame_sha256: annotations.sampling_frame_sha256, annotation_display_version: annotations.annotation_display_version, annotation_display_contract_sha256: annotations.annotation_display_contract_sha256, natural_rgb_preview_sha256: old.natural_rgb_preview_sha256, requires_visual_rereview: false };
      setDirty(); render();
    });
    $("save").addEventListener("click", save);
    document.addEventListener("keydown", (event) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName) && !(event.ctrlKey || event.metaKey)) return;
      const key = event.key.toLowerCase();
      if (event.key === "ArrowLeft") move(-1); else if (event.key === "ArrowRight") move(1);
      else if ({ n: "natural_rgb", f: "false_colour", p: "pca", i: "stored_index", b: "support_outline" }[key]) { $("layer").value = { n: "natural_rgb", f: "false_colour", p: "pca", i: "stored_index", b: "support_outline" }[key]; loadImage(false); }
      else if (["2", "4", "8"].includes(key)) { viewerState = setZoom(viewerState, Number(key)); render(); }
      else if (key === "x") { viewerState = setZoom(viewerState, 16); render(); }
      else if (key === "0") { viewerState = reset(viewerState); render(); }
      else if (key === "g") { $("grid").checked = !$("grid").checked; renderCanvases(); }
      else if ((event.ctrlKey || event.metaKey) && key === "s") { event.preventDefault(); save(); }
    });
    window.addEventListener("beforeunload", (event) => { if (dirty) { event.preventDefault(); event.returnValue = ""; } });
  }
  async function init() {
    [samples, manifest, schema, annotations] = await Promise.all([
      fetch("/api/samples").then((response) => response.json()), fetch("/api/manifest").then((response) => response.json()),
      fetch("/api/schema").then((response) => response.json()), fetch("/api/annotations").then((response) => response.json()),
    ]);
    if (samples.length !== 800 || new Set(samples.map((item) => item.sampling_frame)).size !== 1 || samples[0].sampling_frame !== "main" || schema.reserve_exposed) throw new Error("Expected exact frozen 800-point MAIN frame with reserve locked");
    if (schema.default_layer !== "natural_rgb" || schema.display_version !== annotations.annotation_display_version) throw new Error("Natural RGB display provenance mismatch");
    buildChoices(); bindControls(); bindViewerEvents();
    for (const value of ["", ...new Set(samples.map((item) => item.cube_id))]) { const option = document.createElement("option"); option.value = value; option.textContent = value || "All cubes"; $("cubeFilter").appendChild(option); }
    for (const value of ["", ...new Set(samples.map((item) => item.cube_evaluation_role))]) { const option = document.createElement("option"); option.value = value; option.textContent = value ? value.replaceAll("_", " ") : "All roles"; $("roleFilter").appendChild(option); }
    visible = samples;
    loadImage(true);
  }
  root.__field2ViewerDebug = {
    getState: () => JSON.parse(JSON.stringify(viewerState)),
    getAnnotation: () => JSON.parse(JSON.stringify(currentRecord())),
    eventTargets: () => ({ viewer: true, controlsShieldPointerEvents: getComputedStyle(document.querySelector(".controls-shield")).pointerEvents }),
  };
  init().catch((error) => setStatus(String(error), true));
}(typeof window !== "undefined" ? window : globalThis));
