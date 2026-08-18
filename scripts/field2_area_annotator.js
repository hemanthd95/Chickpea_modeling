(function (root) {
  "use strict";

  const COLORS = {
    research_crop_area: "#22c55e", alley: "#f97316",
    outside_research_field: "#3b82f6", uncertain_boundary: "#a855f7",
  };

  const clone = (value) => JSON.parse(JSON.stringify(value));

  function distanceSquared(first, second) {
    return (first.x - second.x) ** 2 + (first.y - second.y) ** 2;
  }

  function distinctVertexCount(vertices, epsilon = 1e-9) {
    const distinct = [];
    for (const point of vertices) {
      if (!distinct.some((seen) => Math.abs(point.x - seen.x) <= epsilon && Math.abs(point.y - seen.y) <= epsilon)) distinct.push(point);
    }
    return distinct.length;
  }

  function clipPolygonToBounds(vertices, width, height) {
    let points = clone(vertices);
    function clip(pointsIn, inside, intersect) {
      if (!pointsIn.length) return [];
      const output = []; let previous = pointsIn[pointsIn.length - 1];
      for (const current of pointsIn) {
        const currentInside = inside(current), previousInside = inside(previous);
        if (currentInside) { if (!previousInside) output.push(intersect(previous, current)); output.push(clone(current)); }
        else if (previousInside) output.push(intersect(previous, current));
        previous = current;
      }
      return output;
    }
    const vertical = (x) => (a, b) => ({x, y: a.y + (b.y - a.y) * (x - a.x) / (b.x - a.x)});
    const horizontal = (y) => (a, b) => ({x: a.x + (b.x - a.x) * (y - a.y) / (b.y - a.y), y});
    points = clip(points, (p) => p.x >= 0, vertical(0));
    points = clip(points, (p) => p.x <= width, vertical(width));
    points = clip(points, (p) => p.y >= 0, horizontal(0));
    return clip(points, (p) => p.y <= height, horizontal(height));
  }

  function polygonStageCounts(vertices, width, height) {
    const submitted = clone(vertices), clipped = submitted.length >= 3 ? clipPolygonToBounds(submitted, width, height) : submitted;
    return {
      displayed_vertex_count: submitted.length,
      submitted_vertex_count: submitted.length,
      distinct_pixel_vertex_count: distinctVertexCount(submitted),
      clipped_vertex_count: clipped.length,
    };
  }

  function imagePositionFromCanvas(position, currentView, zoom, canvasSize) {
    return {
      x: currentView.center.x + (position.x - canvasSize.width / 2) / zoom,
      y: currentView.center.y + (position.y - canvasSize.height / 2) / zoom,
    };
  }

  function canSwitchDraftZone(draftZone, vertexCount, nextZone) {
    return !vertexCount || !draftZone || draftZone === nextZone;
  }

  function cancelledDraftState() { return {zone_type: "", vertices_pixel: []}; }

  function polygonFromDraft(vertices, zoneType, cubeIdentifier, polygonIdentifier, width, height) {
    const submitted = clone(vertices), counts = polygonStageCounts(submitted, width, height);
    if (counts.distinct_pixel_vertex_count < 3) return {polygon: null, counts};
    return {
      polygon: {polygon_id: polygonIdentifier || `${cubeIdentifier}-polygon-${Date.now()}`, zone_type: zoneType, vertices_pixel: submitted},
      counts,
    };
  }

  function pointInPolygon(point, vertices) {
    let inside = false;
    for (let first = 0, second = vertices.length - 1; first < vertices.length; second = first++) {
      const a = vertices[first], b = vertices[second];
      const crosses = ((a.y > point.y) !== (b.y > point.y)) &&
        point.x < (b.x - a.x) * (point.y - a.y) / ((b.y - a.y) || Number.EPSILON) + a.x;
      if (crosses) inside = !inside;
    }
    return inside;
  }

  function orientation(a, b, c) {
    const value = (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y);
    return Math.abs(value) < 1e-9 ? 0 : (value > 0 ? 1 : 2);
  }

  function onSegment(a, b, c) {
    return b.x >= Math.min(a.x, c.x) - 1e-9 && b.x <= Math.max(a.x, c.x) + 1e-9 &&
      b.y >= Math.min(a.y, c.y) - 1e-9 && b.y <= Math.max(a.y, c.y) + 1e-9;
  }

  function segmentsIntersect(a, b, c, d) {
    const values = [orientation(a, b, c), orientation(a, b, d), orientation(c, d, a), orientation(c, d, b)];
    if (values[0] !== values[1] && values[2] !== values[3]) return true;
    return (values[0] === 0 && onSegment(a, c, b)) || (values[1] === 0 && onSegment(a, d, b)) ||
      (values[2] === 0 && onSegment(c, a, d)) || (values[3] === 0 && onSegment(c, b, d));
  }

  function selfIntersects(vertices) {
    if (vertices.length < 4) return false;
    for (let first = 0; first < vertices.length; first += 1) {
      const firstNext = (first + 1) % vertices.length;
      for (let second = first + 1; second < vertices.length; second += 1) {
        const secondNext = (second + 1) % vertices.length;
        if (first === second || firstNext === second || secondNext === first) continue;
        if (segmentsIntersect(vertices[first], vertices[firstNext], vertices[second], vertices[secondNext])) return true;
      }
    }
    return false;
  }

  function polygonsOverlap(first, second) {
    if (first.vertices_pixel.some((point) => pointInPolygon(point, second.vertices_pixel))) return true;
    if (second.vertices_pixel.some((point) => pointInPolygon(point, first.vertices_pixel))) return true;
    for (let i = 0; i < first.vertices_pixel.length; i += 1) {
      const a = first.vertices_pixel[i], b = first.vertices_pixel[(i + 1) % first.vertices_pixel.length];
      for (let j = 0; j < second.vertices_pixel.length; j += 1) {
        const c = second.vertices_pixel[j], d = second.vertices_pixel[(j + 1) % second.vertices_pixel.length];
        if (segmentsIntersect(a, b, c, d)) return true;
      }
    }
    return false;
  }

  function addPolygon(polygons, polygon) { return [...clone(polygons), clone(polygon)]; }
  function deletePolygon(polygons, polygonId) { return clone(polygons).filter((item) => item.polygon_id !== polygonId); }
  function moveVertex(polygons, polygonId, vertexIndex, point) {
    const result = clone(polygons);
    const polygon = result.find((item) => item.polygon_id === polygonId);
    polygon.vertices_pixel[vertexIndex] = clone(point); delete polygon.vertices_geospatial;
    return result;
  }
  function insertVertex(polygons, polygonId, afterIndex, point) {
    const result = clone(polygons);
    const polygon = result.find((item) => item.polygon_id === polygonId);
    polygon.vertices_pixel.splice(afterIndex + 1, 0, clone(point)); delete polygon.vertices_geospatial;
    return result;
  }
  function deleteVertex(polygons, polygonId, vertexIndex) {
    const result = clone(polygons);
    const polygon = result.find((item) => item.polygon_id === polygonId);
    if (polygon.vertices_pixel.length <= 3) return result;
    polygon.vertices_pixel.splice(vertexIndex, 1); delete polygon.vertices_geospatial;
    return result;
  }
  function movePolygon(polygons, polygonId, delta) {
    const result = clone(polygons);
    const polygon = result.find((item) => item.polygon_id === polygonId);
    polygon.vertices_pixel = polygon.vertices_pixel.map((point) => ({x: point.x + delta.x, y: point.y + delta.y}));
    delete polygon.vertices_geospatial; return result;
  }
  function history(initial) { return {past: [], present: clone(initial), future: []}; }
  function historyApply(value, next) { return {past: [...value.past, clone(value.present)], present: clone(next), future: []}; }
  function historyUndo(value) {
    if (!value.past.length) return value;
    return {past: value.past.slice(0, -1), present: clone(value.past[value.past.length - 1]), future: [clone(value.present), ...value.future]};
  }
  function historyRedo(value) {
    if (!value.future.length) return value;
    return {past: [...value.past, clone(value.present)], present: clone(value.future[0]), future: value.future.slice(1)};
  }

  const Model = {addPolygon, deletePolygon, moveVertex, insertVertex, deleteVertex, movePolygon, history, historyApply, historyUndo, historyRedo, selfIntersects, polygonsOverlap, pointInPolygon, distinctVertexCount, clipPolygonToBounds, polygonStageCounts, imagePositionFromCanvas, canSwitchDraftZone, cancelledDraftState, polygonFromDraft};
  if (typeof module !== "undefined" && module.exports) module.exports = Model;
  root.Field2AreaEditorModel = Model;
  if (typeof document === "undefined") return;

  const $ = (id) => document.getElementById(id);
  const canvas = $("canvas"), context = canvas.getContext("2d");
  let manifest = [], schema = {}, state = {}, cubeIndex = 0, image = new Image(), dirty = false;
  let layer = "natural_rgb", selectedZone = "research_crop_area", tool = "pan", selectedPolygon = "";
  let drawing = [], drawingRedo = [], drawingZone = "", hoverPoint = null, pointerAction = null, opacity = .25;
  let view = {zoom: 0, center: {x: 0, y: 0}};
  const histories = new Map();
  const visibleZones = new Set(Object.keys(COLORS));

  function cubeId() { return manifest[cubeIndex].cube_id; }
  function record() { return state.annotations[cubeId()]; }
  function dimensions() { return {width: manifest[cubeIndex].width, height: manifest[cubeIndex].height}; }
  function effectiveCoverageMode(item) { return item.polygons.length ? "mixed_manual_boundaries" : item.coverage_mode; }
  function historyForCube() {
    if (!histories.has(cubeId())) histories.set(cubeId(), history(record().polygons));
    return histories.get(cubeId());
  }
  function syncHistory(value) { histories.set(cubeId(), value); record().polygons = clone(value.present); }
  function setStatus(message, bad = false) { $("status").textContent = message; $("status").className = bad ? "warning" : ""; }
  function markDirty() {
    dirty = true; record().reviewed = false; record().review_timestamp = "";
    setStatus("UNSAVED area changes — use Save all.", true); renderForm();
  }
  function mutatePolygons(next) { syncHistory(historyApply(historyForCube(), next)); markDirty(); render(); }
  function fitZoom() {
    const step = layer === "natural_rgb" ? 1 : manifest[cubeIndex].preview_step;
    return Math.min(canvas.width / (image.naturalWidth * step), canvas.height / (image.naturalHeight * step));
  }
  function effectiveZoom() { return view.zoom > 0 ? view.zoom : fitZoom(); }
  function resetView() { const size = dimensions(); view = {zoom: 0, center: {x: size.width / 2, y: size.height / 2}}; render(); }
  function transform() {
    const zoom = effectiveZoom();
    return {zoom, x: (value) => canvas.width / 2 + (value - view.center.x) * zoom, y: (value) => canvas.height / 2 + (value - view.center.y) * zoom};
  }
  function eventPosition(event) {
    const rect = canvas.getBoundingClientRect();
    return {x: (event.clientX - rect.left) * canvas.width / rect.width, y: (event.clientY - rect.top) * canvas.height / rect.height};
  }
  function fullPosition(event) {
    const position = eventPosition(event), map = transform();
    return imagePositionFromCanvas(position, view, map.zoom, {width: canvas.width, height: canvas.height});
  }
  function drawPolygon(polygon, selected = false) {
    if (!visibleZones.has(polygon.zone_type)) return;
    const map = transform(), vertices = polygon.vertices_pixel;
    context.save(); context.strokeStyle = COLORS[polygon.zone_type]; context.fillStyle = COLORS[polygon.zone_type];
    context.globalAlpha = opacity; context.lineWidth = selected ? 5 : 3; context.beginPath();
    context.moveTo(map.x(vertices[0].x), map.y(vertices[0].y));
    for (const point of vertices.slice(1)) context.lineTo(map.x(point.x), map.y(point.y));
    context.closePath(); context.fill(); context.globalAlpha = 1; context.stroke();
    if (selected) for (const point of vertices) { context.beginPath(); context.arc(map.x(point.x), map.y(point.y), 5, 0, Math.PI * 2); context.fill(); }
    context.restore();
  }
  function render() {
    if (!image.complete || !image.naturalWidth) return;
    const step = layer === "natural_rgb" ? 1 : manifest[cubeIndex].preview_step, map = transform();
    const scale = map.zoom * step;
    context.clearRect(0, 0, canvas.width, canvas.height); context.fillStyle = "#000"; context.fillRect(0, 0, canvas.width, canvas.height);
    context.imageSmoothingEnabled = map.zoom < 2;
    context.drawImage(image, canvas.width / 2 - (view.center.x / step) * scale, canvas.height / 2 - (view.center.y / step) * scale, image.naturalWidth * scale, image.naturalHeight * scale);
    for (const polygon of record().polygons) drawPolygon(polygon, polygon.polygon_id === selectedPolygon);
    if (drawing.length) {
      const points = hoverPoint ? [...drawing, hoverPoint] : drawing;
      context.save(); context.strokeStyle = COLORS[drawingZone || selectedZone]; context.lineWidth = 2; context.beginPath();
      context.moveTo(map.x(points[0].x), map.y(points[0].y)); for (const point of points.slice(1)) context.lineTo(map.x(point.x), map.y(point.y)); context.stroke();
      for (const point of drawing) { context.beginPath(); context.arc(map.x(point.x), map.y(point.y), 4, 0, Math.PI * 2); context.fillStyle = COLORS[drawingZone || selectedZone]; context.fill(); }
      context.restore();
    }
    renderAudit();
  }
  function renderAudit() {
    const polygons = record().polygons;
    const size = dimensions();
    const selfCount = polygons.filter((item) => selfIntersects(item.vertices_pixel)).length;
    const outsideCount = polygons.filter((item) => item.vertices_pixel.some((point) => point.x < 0 || point.x > size.width || point.y < 0 || point.y > size.height)).length;
    let overlaps = 0;
    for (let first = 0; first < polygons.length; first += 1) for (let second = first + 1; second < polygons.length; second += 1) {
      if (polygons[first].zone_type !== polygons[second].zone_type && polygonsOverlap(polygons[first], polygons[second])) overlaps += 1;
    }
    const coverageWarning = polygons.length && record().coverage_mode !== "mixed_manual_boundaries";
    $("audit").textContent = `${polygons.length} polygons · ${selfCount} self-intersection warnings · ${overlaps} automatically resolved overlap warnings · ${outsideCount} beyond-raster warnings${outsideCount ? " (operational geometry is clipped; raw vertices are preserved)" : ""}${coverageWarning ? " · effective operational mode is mixed_manual_boundaries" : ""}`;
    $("audit").className = selfCount || overlaps || outsideCount || coverageWarning ? "help warning" : "help ok";
    const details = [];
    for (let first = 0; first < polygons.length; first += 1) for (let second = first + 1; second < polygons.length; second += 1) {
      if (polygons[first].zone_type !== polygons[second].zone_type && polygonsOverlap(polygons[first], polygons[second])) {
        details.push(`${polygons[first].polygon_id} (${polygons[first].zone_type}) ↔ ${polygons[second].polygon_id} (${polygons[second].zone_type})`);
      }
    }
    $("auditDetails").textContent = details.length ? `Raw overlaps retained for transparency; operational pixels use outside > alley > uncertain > research crop:\n${details.join("\n")}` : "";
  }
  function selectAt(point) {
    selectedPolygon = "";
    for (const polygon of [...record().polygons].reverse()) if (pointInPolygon(point, polygon.vertices_pixel)) { selectedPolygon = polygon.polygon_id; break; }
    render(); return record().polygons.find((item) => item.polygon_id === selectedPolygon);
  }
  function nearestVertex(polygon, point) {
    let best = -1, distance = Infinity;
    polygon.vertices_pixel.forEach((vertex, index) => { const candidate = distanceSquared(vertex, point); if (candidate < distance) {distance = candidate; best = index;} });
    return {index: best, distance};
  }
  function nearestEdge(polygon, point) {
    let best = 0, distance = Infinity;
    polygon.vertices_pixel.forEach((first, index) => {
      const second = polygon.vertices_pixel[(index + 1) % polygon.vertices_pixel.length];
      const dx = second.x - first.x, dy = second.y - first.y;
      const t = Math.max(0, Math.min(1, ((point.x - first.x) * dx + (point.y - first.y) * dy) / (dx * dx + dy * dy || 1)));
      const candidate = distanceSquared(point, {x: first.x + t * dx, y: first.y + t * dy});
      if (candidate < distance) {distance = candidate; best = index;}
    });
    return best;
  }
  function finishPolygon(trigger = "finish_button") {
    const size = dimensions(), result = polygonFromDraft(drawing, drawingZone || selectedZone, cubeId(), "", size.width, size.height), counts = result.counts;
    if (!result.polygon) {
      const detail = Object.entries(counts).map(([key, value]) => `${key}=${value}`).join("; ");
      return setStatus(`Polygon finalization failed at submitted_geometry_validation (${trigger}): ${detail}. At least three distinct submitted points are required; no vertices were discarded.`, true);
    }
    const polygon = result.polygon;
    mutatePolygons(addPolygon(record().polygons, polygon)); selectedPolygon = polygon.polygon_id;
    drawing = []; drawingRedo = []; drawingZone = ""; hoverPoint = null; tool = "edit"; updateTool();
    if (counts.clipped_vertex_count < counts.submitted_vertex_count) setStatus(`Polygon retained all ${counts.submitted_vertex_count} original vertices; operational raster clipping has ${counts.clipped_vertex_count} boundary vertices. Save all to persist.`, true);
  }
  function updateTool() {
    $("toolStatus").textContent = `Tool: ${tool.replaceAll("_", " ")}`;
    for (const id of ["draw", "edit", "insert", "deleteVertex", "movePolygon", "pan"]) $(id).classList.toggle("active", id === tool || (id === "deleteVertex" && tool === "delete_vertex") || (id === "movePolygon" && tool === "move_polygon"));
  }
  function setTool(name) {
    tool = name; hoverPoint = null; updateTool(); render();
    if (drawing.length && name !== "draw") setStatus(`Draft preserved with ${drawing.length} visible vertices. Return to Draw, then Finish or Cancel current polygon.`, true);
  }
  function cancelDrawing() {
    if (!drawing.length) return setStatus("No current polygon to cancel.", true);
    const cancelled = cancelledDraftState(); drawing = cancelled.vertices_pixel; drawingRedo = []; drawingZone = cancelled.zone_type; hoverPoint = null; pointerAction = null; render();
    setStatus("Current unsaved polygon cancelled explicitly. Existing saved polygons were not changed.");
  }
  function loadImage(reset = false) {
    layer = $("layer").value; image = new Image();
    image.onload = () => { if (reset) resetView(); else render(); renderForm(); };
    image.onerror = () => setStatus("Prediction-free display layer failed to load.", true);
    image.src = `/layers/${cubeId()}/${layer}.png?v=${Date.now()}`;
  }
  function renderForm() {
    const item = record();
    $("coverage").value = item.coverage_mode; $("confidence").value = item.confidence;
    const effectiveMode = effectiveCoverageMode(item);
    $("effectiveMode").textContent = item.polygons.length && item.coverage_mode !== "mixed_manual_boundaries"
      ? `Raw coverage mode: ${item.coverage_mode || "blank"}. Effective operational mode: mixed_manual_boundaries (all ${item.polygons.length} polygons preserved).`
      : `Effective operational mode: ${effectiveMode || "unassigned"}.`;
    $("reviewer").value = item.reviewer_identifier; $("notes").value = item.investigator_notes;
    $("review").textContent = item.reviewed ? "Reviewed ✓" : "Mark cube reviewed";
    $("unassignedStatus").textContent = item.treat_unassigned_valid_support_as_outside ? "Explicit action enabled: unassigned valid support will be outside field at freeze." : "Not enabled; unassigned support remains unassigned.";
    const reviewed = Object.values(state.annotations).filter((value) => value.reviewed).length;
    $("progress").textContent = `${cubeIndex + 1}/40 cubes · ${reviewed}/40 reviewed · revision ${state.revision}`;
    for (const button of $("zoneButtons").querySelectorAll("button")) button.setAttribute("aria-pressed", String(button.dataset.zone === selectedZone));
    renderAudit();
  }
  function updateRecord(field, value) { record()[field] = value; markDirty(); render(); }
  function canNavigate() { if (!drawing.length) return true; setStatus(`Navigation blocked: unfinished ${drawingZone} polygon has ${drawing.length} vertices. Finish or Cancel current polygon first.`, true); return false; }
  function navigate(delta) {
    if (!canNavigate()) return; cubeIndex = Math.max(0, Math.min(manifest.length - 1, cubeIndex + delta));
    $("cube").value = cubeId(); selectedPolygon = ""; loadImage(true);
  }
  async function save() {
    if (drawing.length) return setStatus(`Save blocked: unfinished ${drawingZone} polygon has ${drawing.length} vertices. Finish or Cancel current polygon first.`, true);
    const response = await fetch("/api/annotations", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(state)});
    if (!response.ok) return setStatus(await response.text(), true);
    const result = await response.json(); state.revision = result.revision; dirty = false;
    setStatus(`SAVED revision ${result.revision}: ${result.reviewed}/40 reviewed, ${result.polygons} polygons, ${result.vertices} vertices.`); renderForm();
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    const point = fullPosition(event), screen = eventPosition(event);
    try { canvas.setPointerCapture(event.pointerId); } catch (_) {}
    if (tool === "draw") { pointerAction = {kind: "draw", start: screen}; return; }
    if (tool === "pan") { pointerAction = {kind: "pan", last: screen}; return; }
    const polygon = selectAt(point);
    if (!polygon) return;
    if (tool === "insert") { mutatePolygons(insertVertex(record().polygons, polygon.polygon_id, nearestEdge(polygon, point), point)); return; }
    if (tool === "delete_vertex") {
      const nearest = nearestVertex(polygon, point); if (nearest.distance <= (12 / effectiveZoom()) ** 2) mutatePolygons(deleteVertex(record().polygons, polygon.polygon_id, nearest.index)); return;
    }
    if (tool === "edit") {
      const nearest = nearestVertex(polygon, point); if (nearest.distance <= (12 / effectiveZoom()) ** 2) pointerAction = {kind: "edit", polygonId: polygon.polygon_id, vertexIndex: nearest.index, before: clone(record().polygons)};
    } else if (tool === "move_polygon") pointerAction = {kind: "move", polygonId: polygon.polygon_id, last: point, before: clone(record().polygons)};
  });
  canvas.addEventListener("pointermove", (event) => {
    const point = fullPosition(event), screen = eventPosition(event); hoverPoint = point;
    if (!pointerAction) return render();
    if (pointerAction.kind === "pan") {
      const dx = screen.x - pointerAction.last.x, dy = screen.y - pointerAction.last.y;
      view.center.x -= dx / effectiveZoom(); view.center.y -= dy / effectiveZoom(); pointerAction.last = screen; render();
    } else if (pointerAction.kind === "edit") {
      record().polygons = moveVertex(record().polygons, pointerAction.polygonId, pointerAction.vertexIndex, point); render();
    } else if (pointerAction.kind === "move") {
      const delta = {x: point.x - pointerAction.last.x, y: point.y - pointerAction.last.y};
      record().polygons = movePolygon(record().polygons, pointerAction.polygonId, delta); pointerAction.last = point; render();
    }
  });
  canvas.addEventListener("pointerup", (event) => {
    if (!pointerAction) return;
    if (["edit", "move"].includes(pointerAction.kind)) { syncHistory(historyApply({past: historyForCube().past, present: pointerAction.before, future: []}, record().polygons)); markDirty(); }
    pointerAction = null; render();
  });
  canvas.addEventListener("pointercancel", () => { pointerAction = null; });
  canvas.addEventListener("click", (event) => {
    if (tool !== "draw" || event.button !== 0) return;
    event.preventDefault();
    if (event.detail === 2) return finishPolygon("double_click");
    if (event.detail !== 1) return;
    if (!drawing.length) drawingZone = selectedZone;
    drawing.push(fullPosition(event)); drawingRedo = [];
    setStatus(`Unfinished ${drawingZone} polygon: ${drawing.length} displayed vertices. Finish, double-click, Enter, or Cancel current polygon.`, true);
    render();
  });
  canvas.addEventListener("dblclick", (event) => event.preventDefault());
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault(); const position = eventPosition(event), before = fullPosition(event), old = effectiveZoom();
    const next = Math.max(.05, Math.min(32, old * Math.exp(-event.deltaY * .0015)));
    view.zoom = next; view.center.x = before.x - (position.x - canvas.width / 2) / next; view.center.y = before.y - (position.y - canvas.height / 2) / next; render();
  }, {passive: false});

  $("draw").onclick = () => setTool("draw"); $("finish").onclick = () => finishPolygon("finish_button"); $("cancelDrawing").onclick = cancelDrawing; $("edit").onclick = () => setTool("edit");
  $("insert").onclick = () => setTool("insert"); $("deleteVertex").onclick = () => setTool("delete_vertex"); $("movePolygon").onclick = () => setTool("move_polygon"); $("pan").onclick = () => setTool("pan");
  function removeSelected() { if (!selectedPolygon) return; mutatePolygons(deletePolygon(record().polygons, selectedPolygon)); selectedPolygon = ""; }
  $("deletePolygon").onclick = removeSelected; $("clearSelected").onclick = removeSelected;
  $("undo").onclick = () => {
    if (drawing.length) {drawingRedo.push(drawing.pop()); setStatus(`Draft undo: ${drawing.length} vertices remain.`, true); return render();}
    const value = historyUndo(historyForCube()); if (value !== historyForCube()) {syncHistory(value); markDirty(); render();}
  };
  $("redo").onclick = () => {
    if (drawingRedo.length) {drawing.push(drawingRedo.pop()); setStatus(`Draft redo: ${drawing.length} vertices.`, true); return render();}
    const value = historyRedo(historyForCube()); if (value !== historyForCube()) {syncHistory(value); markDirty(); render();}
  };
  $("previous").onclick = () => navigate(-1); $("next").onclick = () => navigate(1);
  $("cube").onchange = () => { if (!canNavigate()) {$("cube").value = cubeId(); return;} cubeIndex = manifest.findIndex((item) => item.cube_id === $("cube").value); selectedPolygon = ""; loadImage(true); };
  $("layer").onchange = () => loadImage(false); $("resetView").onclick = resetView;
  for (const button of document.querySelectorAll(".zoom")) button.onclick = () => {view.zoom = Number(button.dataset.zoom); render();};
  $("opacity").oninput = () => {opacity = Number($("opacity").value); render();};
  $("coverage").onchange = () => updateRecord("coverage_mode", $("coverage").value);
  $("confidence").onchange = () => updateRecord("confidence", $("confidence").value);
  $("reviewer").onchange = () => updateRecord("reviewer_identifier", $("reviewer").value.trim());
  $("notes").onchange = () => updateRecord("investigator_notes", $("notes").value);
  $("unassignedOutside").onclick = () => {
    if (record().coverage_mode !== "mixed_manual_boundaries") return setStatus("Choose mixed_manual_boundaries before applying the explicit unassigned-support rule.", true);
    if (!record().treat_unassigned_valid_support_as_outside && !confirm("Deliberately treat all unassigned valid support as outside the research field?")) return;
    updateRecord("treat_unassigned_valid_support_as_outside", !record().treat_unassigned_valid_support_as_outside);
  };
  $("review").onclick = () => {
    const item = record(), polygons = item.polygons;
    const effectiveMode = effectiveCoverageMode(item);
    if (!effectiveMode || !item.confidence) return setStatus("An effective coverage mode and confidence are required before review.", true);
    if (effectiveMode === "mixed_manual_boundaries" && !polygons.length) return setStatus("Mixed coverage requires at least one polygon.", true);
    item.reviewed = true; item.review_timestamp = new Date().toISOString(); dirty = true; renderForm(); setStatus("Cube marked reviewed; use Save all.", true);
  };
  $("save").onclick = save;
  document.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && tool === "draw") {event.preventDefault(); finishPolygon("enter_key");}
    if (event.key === "Escape" && drawing.length) {event.preventDefault(); cancelDrawing();}
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {event.preventDefault(); $(event.shiftKey ? "redo" : "undo").click();}
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {event.preventDefault(); save();}
  });
  window.addEventListener("beforeunload", (event) => {if (dirty || drawing.length) {event.preventDefault(); event.returnValue = "";}});

  async function init() {
    [manifest, schema, state] = await Promise.all([
      fetch("/api/manifest").then((response) => response.json()), fetch("/api/schema").then((response) => response.json()), fetch("/api/annotations").then((response) => response.json()),
    ]);
    if (manifest.length !== 40 || schema.reserve_exposed || schema.biological_labels_available || schema.default_layer !== "natural_rgb") throw new Error("Area safety gate failed");
    for (const item of manifest) {const option = document.createElement("option"); option.value = item.cube_id; option.textContent = item.cube_id; $("cube").appendChild(option);}
    for (const mode of schema.coverage_modes) {const option = document.createElement("option"); option.value = mode; option.textContent = mode.replaceAll("_", " "); $("coverage").appendChild(option);}
    for (const value of schema.confidence) {const option = document.createElement("option"); option.value = value; option.textContent = value; $("confidence").appendChild(option);}
    for (const zone of schema.zone_types) {
      const button = document.createElement("button"); button.type = "button"; button.dataset.zone = zone; button.className = `zone-${zone}`; button.textContent = zone.replaceAll("_", " "); button.onclick = () => {
        if (!canSwitchDraftZone(drawingZone, drawing.length, zone)) return setStatus(`Zone switch blocked: the unfinished polygon is ${drawingZone}. Finish or Cancel it before selecting ${zone}; no vertices were discarded.`, true);
        selectedZone = zone; setTool("draw"); renderForm();
      }; $("zoneButtons").appendChild(button);
      const toggle = document.createElement("button"); toggle.type = "button"; toggle.dataset.zone = zone; toggle.className = `zone-${zone}`; toggle.textContent = `✓ ${zone.replaceAll("_", " ")}`; toggle.setAttribute("aria-pressed", "true"); toggle.onclick = () => {if (visibleZones.has(zone)) visibleZones.delete(zone); else visibleZones.add(zone); toggle.setAttribute("aria-pressed", String(visibleZones.has(zone))); toggle.textContent = `${visibleZones.has(zone) ? "✓" : "○"} ${zone.replaceAll("_", " ")}`; render();}; $("zoneToggles").appendChild(toggle);
    }
    updateTool(); loadImage(true);
  }
  root.__field2AreaDebug = {getRecord: () => clone(record()), getView: () => clone(view), getTool: () => tool, getDraft: () => ({zone_type: drawingZone, vertices_pixel: clone(drawing)})};
  init().catch((error) => setStatus(String(error), true));
}(typeof window !== "undefined" ? window : globalThis));
