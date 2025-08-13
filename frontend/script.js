const API_BASE = (window.API_BASE || "http://localhost:8000");

async function postMedia(url, file, extra = {}) {
  const form = new FormData();
  form.append("file", file);
  for (const [k, v] of Object.entries(extra)) form.append(k, v);
  const res = await fetch(url, { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return await res.json();
}

function setResult(el, warnEl, data) {
  if (!data) return;
  const cls = data.is_ai ? "ai" : "real";
  const emoji = data.is_ai ? "⚠️" : "✅";
  el.className = `result ${cls}`;
  el.textContent = `${emoji} ${data.label} — confidence ${(data.confidence * 100).toFixed(1)}%`;
  warnEl.textContent = data.detail || "";
}

function setLoading(btn, loading) {
  btn.disabled = !!loading;
  btn.textContent = loading ? "Analyzing..." : btn.dataset.label;
}

async function handleImage() {
  const file = document.getElementById("imageInput").files[0];
  const btn = document.getElementById("imageBtn");
  const resEl = document.getElementById("imageResult");
  const warnEl = document.getElementById("imageWarn");
  if (!file) return alert("Please choose an image file");
  setLoading(btn, true);
  try {
    const data = await postMedia(`${API_BASE}/predict/image`, file);
    setResult(resEl, warnEl, data);
  } catch (e) {
    resEl.className = "result";
    resEl.textContent = `Error: ${e.message}`;
  } finally {
    setLoading(btn, false);
  }
}

async function handleVideo() {
  const file = document.getElementById("videoInput").files[0];
  const btn = document.getElementById("videoBtn");
  const resEl = document.getElementById("videoResult");
  const warnEl = document.getElementById("videoWarn");
  if (!file) return alert("Please choose a video file");
  setLoading(btn, true);
  try {
    const data = await postMedia(`${API_BASE}/predict/video`, file, { num_frames: 32 });
    setResult(resEl, warnEl, data);
  } catch (e) {
    resEl.className = "result";
    resEl.textContent = `Error: ${e.message}`;
  } finally {
    setLoading(btn, false);
  }
}

window.addEventListener("DOMContentLoaded", () => {
  const imgBtn = document.getElementById("imageBtn");
  imgBtn.dataset.label = "Analyze Image";
  imgBtn.addEventListener("click", handleImage);

  const vidBtn = document.getElementById("videoBtn");
  vidBtn.dataset.label = "Analyze Video";
  vidBtn.addEventListener("click", handleVideo);
});