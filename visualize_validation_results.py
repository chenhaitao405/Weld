#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""将 validate_inference_results.py 产出转为可离线查看的HTML报告。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


STATUS_SUCCESS_CLASS = "成功分类"
STATUS_SUCCESS_DETECT = "成功检出"
STATUS_MISSED = "漏检"
STATUS_FALSE = "误检"

STATUS_ORDER = [
    STATUS_SUCCESS_CLASS,
    STATUS_SUCCESS_DETECT,
    STATUS_MISSED,
    STATUS_FALSE,
]

STATUS_COLORS = {
    STATUS_SUCCESS_CLASS: "#2ecc71",
    STATUS_SUCCESS_DETECT: "#ffa500",
    STATUS_MISSED: "#ff0000",
    STATUS_FALSE: "#8e44ad",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取验证输出目录，生成可离线浏览的HTML报告"
    )
    parser.add_argument("--validation-dir", required=True,
                        help="validate_inference_results.py 的输出目录")
    parser.add_argument("--output-html", help="HTML输出路径，默认 validation_dir/report.html")
    parser.add_argument("--title", default="焊缝缺陷验证可视化",
                        help="HTML标题")
    parser.add_argument("--max-images", type=int,
                        help="仅导出前N张图以减小HTML体积")
    return parser.parse_args()


def load_manifest(validation_dir: Path) -> Dict[str, Any]:
    manifest_path = validation_dir / "data" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"未找到manifest: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_summary(validation_dir: Path) -> Dict[str, Any]:
    summary_path = validation_dir / "metrics_summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_detail(validation_dir: Path, relative_path: str) -> Dict[str, Any]:
    detail_path = validation_dir / relative_path
    if not detail_path.exists():
        raise FileNotFoundError(f"未找到 detail: {detail_path}")
    with open(detail_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_entries(validation_dir: Path,
                  html_dir: Path,
                  manifest: Dict[str, Any],
                  limit: int | None) -> List[Dict[str, Any]]:
    images = manifest.get("images", [])
    if limit is not None:
        images = images[:limit]

    entries: List[Dict[str, Any]] = []
    for idx, item in enumerate(images):
        detail_rel = item.get("detail_path")
        if not detail_rel:
            continue
        detail = read_detail(validation_dir, detail_rel)
        image_rel = item.get("copied_image_path") or item.get("relative_image_path")
        overlay_rel = item.get("overlay_path")
        image_web_path = _to_relative_url(validation_dir, html_dir, image_rel)
        overlay_web_path = _to_relative_url(validation_dir, html_dir, overlay_rel)
        entries.append({
            "id": idx,
            "image_path": image_web_path,
            "overlay_path": overlay_web_path,
            "detail_path": detail_rel,
            "metrics": detail.get("metrics", {}),
            "status_counts": item.get("status_counts", {}),
            "predictions": detail.get("predictions", []),
            "annotations": detail.get("annotations", []),
            "width": detail.get("width"),
            "height": detail.get("height"),
            "image_basename": Path(detail.get("relative_image_path", "")).name,
        })
    return entries


def _to_relative_url(validation_dir: Path, html_dir: Path, rel_path: str | None) -> str | None:
    if not rel_path:
        return None
    abs_path = (validation_dir / rel_path).resolve()
    rel_to_html = os.path.relpath(abs_path, html_dir)
    return Path(rel_to_html).as_posix()


def build_html(title: str,
               entries: List[Dict[str, Any]],
               summary: Dict[str, Any]) -> str:
    data = {
        "title": title,
        "entries": entries,
        "summary": summary,
        "status_order": STATUS_ORDER,
        "status_colors": STATUS_COLORS,
    }
    data_json = json.dumps(data, ensure_ascii=False)

    css = _build_css()
    js = _build_js()
    annotation_filter_html = ''.join([
        f'<label><input type="checkbox" class="annotation-checkbox" data-status="{status}" checked />{status}</label>'
        for status in STATUS_ORDER
    ])
    annotation_filter_modal_html = ''.join([
        f'<label><input type="checkbox" class="annotation-checkbox annotation-checkbox-modal" data-status="{status}" checked />{status}</label>'
        for status in STATUS_ORDER
    ])
    sample_filter_html = ''.join([
        f'<label><input type="checkbox" class="sample-checkbox" data-status="{status}" checked />{status}</label>'
        for status in STATUS_ORDER
    ])

    html = f"""<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\" />
  <title>{title}</title>
  <style>
{css}
  </style>
</head>
<body>
  <div class=\"header\">
    <h1>{title}</h1>
    <div class=\"metrics\" id=\"global-metrics\"></div>
    <div class=\"controls\">
      <label><input type=\"checkbox\" class=\"display-toggle\" data-toggle=\"pred\" checked />显示推理结果</label>
      <label><input type=\"checkbox\" class=\"display-toggle\" data-toggle=\"gt\" checked />显示原始标注</label>
      <label><input type=\"checkbox\" class=\"display-toggle\" data-toggle=\"labels\" checked />显示交并比标签</label>
    </div>
    <div class=\"status-filter\">
      <span>筛选结果标签：</span>
      {annotation_filter_html}
    </div>
    <div class=\"status-filter sample-filter\">
      <span>筛选样本：</span>
      {sample_filter_html}
      <span class=\"sample-counter\" id=\"sample-counter\"></span>
    </div>
  </div>
  <div class=\"legend\" id=\"legend\"></div>
  <div class=\"gallery\" id=\"gallery\"></div>
  <div class=\"modal\" id=\"preview-modal\">
    <div class=\"modal-content\">
      <div class=\"modal-header\">
        <div id=\"modal-title\"></div>
        <div class=\"modal-controls\">
          <div class=\"modal-toggles\">
            <label><input type=\"checkbox\" class=\"display-toggle modal-display\" data-toggle=\"pred\" checked />显示推理结果</label>
            <label><input type=\"checkbox\" class=\"display-toggle modal-display\" data-toggle=\"gt\" checked />显示原始标注</label>
            <label><input type=\"checkbox\" class=\"display-toggle modal-display\" data-toggle=\"labels\" checked />显示交并比标签</label>
          </div>
          <button id=\"modal-close\">关闭</button>
        </div>
      </div>
      <div class=\"modal-filter-bar\">
        <span>筛选结果标签：</span>
        {annotation_filter_modal_html}
      </div>
      <div class=\"modal-body\">
        <div class=\"modal-canvas-wrapper\" id=\"modal-canvas-wrapper\">
          <img id=\"modal-image\" alt=\"preview\" />
          <canvas id=\"modal-pred\"></canvas>
          <canvas id=\"modal-gt\"></canvas>
        </div>
      </div>
    </div>
  </div>
  <script>
    const VIS_DATA = {data_json};
{js}
  </script>
</body>
</html>"""
    return html


def _build_css() -> str:
    return r"""
body {
  font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
  margin: 0;
  padding: 0;
  background: #f5f6fa;
  color: #2c3e50;
}
.header {
  background: #fff;
  padding: 16px 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  position: sticky;
  top: 0;
  z-index: 10;
}
.controls label, .status-filter label {
  margin-right: 16px;
  font-size: 14px;
}
.sample-filter {
  margin-top: 6px;
}
.legend {
  padding: 12px 24px;
  background: #fff;
  border-bottom: 1px solid #e1e5ee;
  font-size: 14px;
}
.legend span {
  margin-right: 18px;
}
.sample-counter {
  font-weight: 600;
  margin-left: 12px;
  color: #34495e;
}
.gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  gap: 16px;
  padding: 16px 24px 48px;
}
.card {
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.08);
  overflow: hidden;
  display: flex;
  flex-direction: column;
  cursor: zoom-in;
}
.card-header {
  padding: 12px 16px;
  border-bottom: 1px solid #f0f0f0;
}
.card-body {
  padding: 12px 16px 16px;
}
.image-stack {
  position: relative;
  width: 100%;
  background: #111;
  border-radius: 6px;
  overflow: hidden;
}
.image-stack img, .image-stack canvas {
  width: 100%;
  display: block;
}
.image-stack canvas {
  position: absolute;
  top: 0;
  left: 0;
}
.metrics-line {
  font-size: 13px;
  margin-top: 4px;
}
.badge {
  display: inline-block;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 12px;
  color: #fff;
  margin-right: 6px;
}
.status-filter {
  margin-top: 12px;
}
@media (max-width: 600px) {
  .controls label, .status-filter label {
    display: block;
    margin: 6px 0;
  }
}
.modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0,0,0,0.6);
  display: none;
  align-items: center;
  justify-content: center;
  padding: 24px;
  z-index: 99;
}
.modal.visible {
  display: flex;
}
.modal-content {
  background: #fff;
  width: 90%;
  max-width: 1400px;
  max-height: 100%;
  border-radius: 10px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.modal-header {
  padding: 12px 16px;
  border-bottom: 1px solid #eee;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.modal-controls label {
  margin-right: 12px;
}
.modal-toggles {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-right: 12px;
}
.modal-filter-bar {
  color: #ecf0f1;
  padding: 10px 18px;
  font-size: 14px;
  background: rgba(24,24,24,0.95);
  position: sticky;
  top: 0;
  z-index: 3;
}
.modal-filter-bar label {
  margin-right: 18px;
}
.modal-filter-bar span {
  margin-right: 12px;
}
.modal-body {
  padding: 0;
  flex: 1;
  overflow: auto;
  background: #181818;
}
.modal-canvas-wrapper {
  position: relative;
  width: 100%;
  height: 100%;
  transform-origin: top left;
  cursor: grab;
}
.modal-canvas-wrapper img,
.modal-canvas-wrapper canvas {
  display: block;
  width: 100%;
}
.modal-canvas-wrapper canvas {
  position: absolute;
  top: 0;
  left: 0;
}
.modal-close {
  cursor: pointer;
}
"""


def _build_js() -> str:
    return r"""
const STATUS_COLORS = VIS_DATA.status_colors;
const STATUS_ORDER = VIS_DATA.status_order;
const STATUS_SUCCESS_CLASS = STATUS_ORDER[0];
const STATUS_SUCCESS_DETECT = STATUS_ORDER[1];
const STATUS_MISSED = STATUS_ORDER[2];
const STATUS_FALSE = STATUS_ORDER[3];

const state = {
  showPred: true,
  showGT: true,
  showLabels: true,
  statusVisible: new Set(STATUS_ORDER),
  sampleFilter: new Set(STATUS_ORDER),
};

let modalEntry = null;
const modalElements = {};
const modalState = {
  scale: 1,
  offsetX: 0,
  offsetY: 0,
  dragging: false,
  startX: 0,
  startY: 0,
};
const totalEntries = (VIS_DATA.entries || []).length;

document.addEventListener('DOMContentLoaded', () => {
  initLegend();
  initGlobalMetrics();
  initControls();
  initModal();
  renderGallery();
});

function initLegend() {
  const container = document.getElementById('legend');
  container.innerHTML = STATUS_ORDER.map(status => {
    return `<span><span class="badge" style="background:${STATUS_COLORS[status]}"></span>${status}</span>`;
  }).join('');
}

function initGlobalMetrics() {
  const metricsBox = document.getElementById('global-metrics');
  const overall = (VIS_DATA.summary && VIS_DATA.summary.overall) || {};
  const metrics = [];
  if (overall.defect_precision != null) {
    metrics.push(`缺陷识别准确率: ${(overall.defect_precision * 100).toFixed(2)}%`);
  }
  if (overall.defect_recall != null) {
    metrics.push(`缺陷识别召回率: ${(overall.defect_recall * 100).toFixed(2)}%`);
  }
  if (overall.classification_accuracy != null) {
    metrics.push(`缺陷分类准确率: ${(overall.classification_accuracy * 100).toFixed(2)}%`);
  }
  if (overall.counts) {
    const counts = overall.counts;
    metrics.push(`预测数: ${counts.predictions || 0}`);
    metrics.push(`标注数: ${counts.ground_truth || 0}`);
  }
  metricsBox.innerHTML = metrics.map(m => `<div class="metrics-line">${m}</div>`).join('');
}

function initControls() {
  document.querySelectorAll('.display-toggle').forEach(cb => {
    cb.addEventListener('change', onDisplayToggleChange);
  });
  document.querySelectorAll('.annotation-checkbox').forEach(cb => {
    cb.addEventListener('change', (e) => {
      const status = e.target.dataset.status;
      if (e.target.checked) {
        state.statusVisible.add(status);
      } else {
        state.statusVisible.delete(status);
      }
      renderGallery();
      refreshModalCanvas();
      syncAnnotationCheckboxes();
    });
  });
  document.querySelectorAll('.sample-checkbox').forEach(cb => {
    cb.addEventListener('change', (e) => {
      const status = e.target.dataset.status;
      if (e.target.checked) {
        state.sampleFilter.add(status);
      } else {
        state.sampleFilter.delete(status);
      }
      renderGallery();
      updateSampleCounter();
    });
  });
  syncDisplayToggles();
  syncAnnotationCheckboxes();
}

function onDisplayToggleChange(e) {
  const target = e.target.dataset.toggle;
  if (!target) return;
  if (target === 'pred') state.showPred = e.target.checked;
  if (target === 'gt') state.showGT = e.target.checked;
  if (target === 'labels') state.showLabels = e.target.checked;
  syncDisplayToggles();
  renderGallery();
  refreshModalCanvas();
}

function renderGallery() {
  const gallery = document.getElementById('gallery');
  gallery.innerHTML = '';
  const filtered = getFilteredEntries();
  filtered.forEach(entry => {
    gallery.appendChild(renderCard(entry));
  });
  updateSampleCounter(filtered.length);
}

function getFilteredEntries() {
  const entries = VIS_DATA.entries || [];
  return entries.filter(entry => shouldDisplayEntry(entry));
}

function shouldDisplayEntry(entry) {
  if (state.sampleFilter.size === STATUS_ORDER.length) {
    return true;
  }
  if (state.sampleFilter.size === 0) {
    return false;
  }
  const counts = entry.status_counts || {};
  for (const status of state.sampleFilter) {
    if ((counts[status] || 0) > 0) {
      return true;
    }
  }
  return false;
}

function renderCard(entry) {
  const card = document.createElement('div');
  card.className = 'card';
  card.addEventListener('click', () => openModal(entry));

  const header = document.createElement('div');
  header.className = 'card-header';
  const title = document.createElement('div');
  title.textContent = entry.image_basename || entry.image_path;
  header.appendChild(title);

  const metricLine = document.createElement('div');
  metricLine.className = 'metrics-line';
  const counts = entry.metrics && entry.metrics.counts || {};
  const detected = (counts[STATUS_SUCCESS_CLASS] || 0) + (counts[STATUS_SUCCESS_DETECT] || 0);
  const precision = entry.metrics && entry.metrics.defect_precision != null ? (entry.metrics.defect_precision * 100).toFixed(1) : 'N/A';
  const recall = entry.metrics && entry.metrics.defect_recall != null ? (entry.metrics.defect_recall * 100).toFixed(1) : 'N/A';
  const clsAcc = entry.metrics && entry.metrics.classification_accuracy != null ? (entry.metrics.classification_accuracy * 100).toFixed(1) : 'N/A';
  metricLine.innerHTML = `命中: ${detected}/${counts.ground_truth || 0} · 精确率: ${precision}% · 召回率: ${recall}% · 分类准确率: ${clsAcc}%`;
  header.appendChild(metricLine);
  card.appendChild(header);

  const body = document.createElement('div');
  body.className = 'card-body';

  const imageStack = document.createElement('div');
  imageStack.className = 'image-stack';
  const img = document.createElement('img');
  img.src = safePath(entry.image_path) || safePath(entry.overlay_path) || '';
  img.alt = entry.image_basename;
  imageStack.appendChild(img);

  const predCanvas = document.createElement('canvas');
  const gtCanvas = document.createElement('canvas');
  imageStack.appendChild(predCanvas);
  imageStack.appendChild(gtCanvas);
  body.appendChild(imageStack);

  img.addEventListener('load', () => {
    prepareCanvases(img, [predCanvas, gtCanvas]);
    renderEntryLayers(entry, img, predCanvas, gtCanvas);
  });

  card.appendChild(body);

  return card;
}

function prepareCanvases(img, canvases) {
  canvases.forEach(canvas => {
    if (!canvas) return;
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.style.width = '100%';
    canvas.style.height = 'auto';
  });
}

function renderEntryLayers(entry, img, predCanvas, gtCanvas) {
  if (!predCanvas || !gtCanvas) return;
  const predCtx = predCanvas.getContext('2d');
  const gtCtx = gtCanvas.getContext('2d');
  predCtx.clearRect(0, 0, predCanvas.width, predCanvas.height);
  gtCtx.clearRect(0, 0, gtCanvas.width, gtCanvas.height);

  predCanvas.style.display = state.showPred ? 'block' : 'none';
  gtCanvas.style.display = state.showGT ? 'block' : 'none';

  if (state.showPred) {
    (entry.predictions || []).forEach(pred => {
      const status = pred.status || STATUS_FALSE;
      if (!statusAllowed(status)) {
        return;
      }
      const color = STATUS_COLORS[status] || '#ffffff';
      drawShape(predCtx, pred, color);
      if (state.showLabels) {
        drawLabel(predCtx, pred, color, status);
      }
    });
  }

  if (state.showGT) {
    (entry.annotations || []).forEach(ann => {
      const status = ann.status || '标注';
      if (!statusAllowed(status)) {
        return;
      }
      const color = ann.status === STATUS_MISSED ? STATUS_COLORS[STATUS_MISSED] : '#3498db';
      drawBox(gtCtx, ann.bbox, color, true);
      if (state.showLabels) {
        drawLabel(gtCtx, ann, color, ann.status ? `${ann.status}` : `标注:${ann.class_name || ''}`);
      }
    });
  }
}

function initModal() {
  modalElements.container = document.getElementById('preview-modal');
  if (!modalElements.container) return;
  modalElements.title = document.getElementById('modal-title');
  modalElements.image = document.getElementById('modal-image');
  modalElements.pred = document.getElementById('modal-pred');
  modalElements.gt = document.getElementById('modal-gt');
  modalElements.wrapper = document.getElementById('modal-canvas-wrapper');
  modalElements.closeBtn = document.getElementById('modal-close');

  if (modalElements.closeBtn) {
    modalElements.closeBtn.addEventListener('click', closeModal);
  }
  modalElements.container.addEventListener('click', (e) => {
    if (e.target === modalElements.container) {
      closeModal();
    }
  });
  if (modalElements.image) {
    modalElements.image.addEventListener('load', () => {
      if (!modalEntry) return;
      prepareCanvases(modalElements.image, [modalElements.pred, modalElements.gt]);
      renderEntryLayers(modalEntry, modalElements.image, modalElements.pred, modalElements.gt);
    });
  }
  if (modalElements.wrapper) {
    modalElements.wrapper.addEventListener('wheel', handleModalWheel, { passive: false });
    modalElements.wrapper.addEventListener('mousedown', startModalDrag);
  }
  window.addEventListener('mousemove', onModalDrag);
  window.addEventListener('mouseup', endModalDrag);
}

function openModal(entry) {
  if (!modalElements.container || !entry) return;
  modalEntry = entry;
  if (modalElements.title) {
    modalElements.title.textContent = entry.image_basename || entry.image_path;
  }
  modalElements.container.classList.add('visible');
  resetModalTransform();
  const src = safePath(entry.image_path) || safePath(entry.overlay_path) || '';
  if (modalElements.image) {
    modalElements.image.src = src;
  }
}

function closeModal() {
  if (!modalElements.container) return;
  modalElements.container.classList.remove('visible');
}

function handleModalWheel(event) {
  if (!modalElements.wrapper) return;
  event.preventDefault();
  const delta = event.deltaY > 0 ? -0.1 : 0.1;
  modalState.scale = clamp(modalState.scale + delta, 0.2, 5);
  updateModalTransform();
}

function startModalDrag(event) {
  if (!modalElements.wrapper) return;
  event.preventDefault();
  modalState.dragging = true;
  modalState.startX = event.clientX - modalState.offsetX;
  modalState.startY = event.clientY - modalState.offsetY;
  modalElements.wrapper.style.cursor = 'grabbing';
}

function onModalDrag(event) {
  if (!modalState.dragging) return;
  modalState.offsetX = event.clientX - modalState.startX;
  modalState.offsetY = event.clientY - modalState.startY;
  updateModalTransform();
}

function endModalDrag() {
  if (!modalState.dragging) return;
  modalState.dragging = false;
  if (modalElements.wrapper) {
    modalElements.wrapper.style.cursor = 'grab';
  }
}

function resetModalTransform() {
  modalState.scale = 1;
  modalState.offsetX = 0;
  modalState.offsetY = 0;
  modalState.dragging = false;
  updateModalTransform();
  if (modalElements.wrapper) {
    modalElements.wrapper.style.cursor = 'grab';
  }
}

function updateModalTransform() {
  if (!modalElements.wrapper) return;
  modalElements.wrapper.style.transform = `translate(${modalState.offsetX}px, ${modalState.offsetY}px) scale(${modalState.scale})`;
}

function drawShape(ctx, data, color) {
  if (data.polygon && data.polygon.length >= 3) {
    drawPolygon(ctx, data.polygon, color);
  } else {
    drawBox(ctx, data.bbox, color);
  }
}

function drawBox(ctx, bbox, color, dashed = false) {
  if (!bbox) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  if (dashed && ctx.setLineDash) {
    ctx.setLineDash([10, 6]);
  }
  ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
  if (dashed && ctx.setLineDash) {
    ctx.setLineDash([]);
  }
}

function drawPolygon(ctx, polygon, color) {
  ctx.beginPath();
  polygon.forEach((pt, idx) => {
    if (idx === 0) ctx.moveTo(pt[0], pt[1]);
    else ctx.lineTo(pt[0], pt[1]);
  });
  ctx.closePath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.globalAlpha = 0.25;
  ctx.fillStyle = color;
  ctx.fill();
  ctx.globalAlpha = 1;
}

function drawLabel(ctx, data, color, statusText) {
  const bbox = data.bbox;
  if (!bbox) return;
  const text = [statusText, data.class_name || '', data.confidence != null ? data.confidence.toFixed(2) : null, data.iou != null ? `IoU:${data.iou.toFixed(2)}` : null]
    .filter(Boolean)
    .join(' | ');
  if (!text) return;
  const x = bbox[0] + 4;
  const y = Math.max(16, bbox[1] + 16);
  ctx.font = '20px sans-serif';
  ctx.fillStyle = 'rgba(0,0,0,0.6)';
  const textWidth = ctx.measureText(text).width;
  ctx.fillRect(x - 2, y - 18, textWidth + 6, 20);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y - 2);
}

function safePath(rel) {
  if (!rel) return '';
  if (rel.startsWith('http://') || rel.startsWith('https://') || rel.startsWith('data:')) {
    return rel;
  }
  return rel.replace(/\\/g, '/');
}

function statusAllowed(status) {
  if (!status) return true;
  if (!STATUS_ORDER.includes(status)) return true;
  return state.statusVisible.has(status);
}

function syncDisplayToggles() {
  document.querySelectorAll('.display-toggle').forEach(cb => {
    const target = cb.dataset.toggle;
    if (target === 'pred') cb.checked = state.showPred;
    if (target === 'gt') cb.checked = state.showGT;
    if (target === 'labels') cb.checked = state.showLabels;
  });
}

function syncAnnotationCheckboxes() {
  document.querySelectorAll('.annotation-checkbox').forEach(cb => {
    const status = cb.dataset.status;
    cb.checked = state.statusVisible.has(status);
  });
}

function refreshModalCanvas() {
  if (!modalEntry || !modalElements.image) return;
  prepareCanvases(modalElements.image, [modalElements.pred, modalElements.gt]);
  renderEntryLayers(modalEntry, modalElements.image, modalElements.pred, modalElements.gt);
}

function updateSampleCounter(currentCount) {
  const counter = document.getElementById('sample-counter');
  if (!counter) return;
  const total = totalEntries || 0;
  const shown = typeof currentCount === 'number' ? currentCount : getFilteredEntries().length;
  counter.textContent = `${shown}/${total}`;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}
"""


def write_html(html_path: Path, content: str):
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(content, encoding="utf-8")


def main():
    args = parse_args()
    validation_dir = Path(args.validation_dir).resolve()
    output_html = Path(args.output_html).resolve() if args.output_html else validation_dir / "report.html"

    manifest = load_manifest(validation_dir)
    summary = load_summary(validation_dir)
    html_dir = output_html.parent
    entries = build_entries(validation_dir, html_dir, manifest, args.max_images)
    if not entries:
        raise RuntimeError("manifest中没有图像条目，无法生成可视化")

    html = build_html(args.title, entries, summary)
    write_html(output_html, html)
    print(f"HTML 已生成：{output_html}")


if __name__ == "__main__":
    main()
