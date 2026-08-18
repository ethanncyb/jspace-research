#!/usr/bin/env python3
"""Local interactive panel for the Gemma 4 Jacobian lens.

    python3 scripts/lens_panel.py --session lens
    # then open http://127.0.0.1:8765

The GPU work happens on Colab; the view renders in your local browser. There is no
tunnel and no public URL -- the transport is `colab exec` against a persistent
session, reusing the ADC auth you already have. The panel owns the CLI, so it can
also stop the session for you when you're done.

Flow per prompt:

    browser  --POST /api/slice-->  this server  --colab exec-->  warm kernel
                                        |                            |
                                   extract tarball  <--base64 stdout--
                                        |
    iframe /vis.html?datapath=/data/<run_id>/  <-- sidecar files served locally

The visualisation itself is jlens's own `mode="fetch"` page, which reads meta.json /
slice.bin / ranks/*.bin from whatever `?datapath=` points at. That page is
slice-independent, so it is fetched and cached once and then repointed per run.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BEGIN = "<<<JLENS_PAYLOAD_BEGIN>>>"
END = "<<<JLENS_PAYLOAD_END>>>"

SLICE_VIS_URL = ("https://raw.githubusercontent.com/anthropics/jacobian-lens/"
                 "main/jlens/data/slice_vis.html")
D3_TAG = ('<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js" '
          'integrity="sha384-CjloA8y00+1SDAUkjs099PVfnY2KmDC2BZnws9kh8D/lX1s46w6EPhpXdqMfjK6i" '
          'crossorigin="anonymous"></script>')

STATE = threading.local()

#: What the server is doing right now, polled by the browser via /api/progress.
#: Without this the panel showed "computing slice" for the whole request, including
#: a ~6 min cold start (lens upload + 10GB weight download), which looks like a hang.
_progress = {"phase": "idle", "detail": ""}


def set_progress(phase: str, detail: str = "") -> None:
    _progress["phase"] = phase
    _progress["detail"] = detail
    if phase != "idle":
        print(f"[panel] {phase}: {detail}" if detail else f"[panel] {phase}",
              file=sys.stderr)


#: One exec at a time -- it's a single kernel. Reentrant on purpose: ensure_booted()
#: holds this across the whole boot, and upload_large() calls exec_code() inside that
#: to reassemble chunks. With a plain Lock the same thread blocks on itself and the
#: panel hangs forever right after the last chunk uploads.
_colab_lock = threading.RLock()


class Config:
    session = "lens"
    timeout = 900
    root = Path(__file__).resolve().parent.parent / "panel"
    service = Path(__file__).resolve().parent / "lens_slice_service.py"
    lens = Path(__file__).resolve().parent.parent / "panel" / "gemma4_12b_qat_lens.pt"
    gpu = "A100"
    #: Preferred source for the lens: the VM pulls it from the Hub instead of us
    #: pushing 649MB up a home uplink. Set hf_repo = "" (--no-hf) to force the push.
    hf_repo = "PxlNexus/gemma4-12b-QAT-Jlens"
    hf_file = "gemma4_12b_qat_lens.pt"
    #: sha256 of the fitted lens, checked on the VM after download. Guards against a
    #: truncated transfer and against the repo silently changing under you.
    hf_sha256 = "38dbf7a1faedde7f550892c2c0f4378409c701c3948f2a4b618fec3c6087045a"


def colab(*args: str, stdin: str | None = None, timeout: int | None = None):
    """Run the colab CLI. `--auth=adc` must precede the subcommand."""
    cmd = ["colab", "--auth=adc", *args]
    return subprocess.run(cmd, input=stdin, capture_output=True, text=True,
                          timeout=timeout or (Config.timeout + 120))


#: `colab upload` reads the whole file, base64-encodes it, and PUTs it as ONE JSON
#: body (colab_cli/contents.py: `payload = {..., "content": b64, "chunk": 1}`), so a
#: large file becomes a >100MB request and the contents server rejects it. Measured
#: on a CPU runtime: 64MB uploads in 19s, 128MB returns 500, 256MB returns 400.
#: 32MB chunks upload in ~6s each, so that is the working size.
CHUNK_BYTES = 32 * 1024 * 1024


def _upload_one(local: Path, remote: str) -> None:
    proc = colab("upload", "-s", Config.session, str(local), remote, timeout=900)
    if proc.returncode != 0 or "Upload failed" in (proc.stdout or ""):
        raise RuntimeError((proc.stdout or proc.stderr).strip()[-400:])


def upload_large(local: Path, remote: str, chunk: int = CHUNK_BYTES) -> None:
    """Upload a file of any size, splitting it if the contents API can't take it.

    Parts are written and uploaded one at a time (never the whole file again on
    disk), concatenated on the VM, then checked by size and sha256 before the
    parts are removed.
    """
    size = local.stat().st_size
    if size <= chunk:
        _upload_one(local, remote)
        return

    n_parts = (size + chunk - 1) // chunk
    digest = hashlib.sha256()
    set_progress("uploading the lens to the new VM",
                 f"{size / 1e6:.0f} MB in {n_parts} chunks, ~{n_parts * 8}s")
    print(f"[panel] uploading {size / 1e6:.0f} MB in {n_parts} chunks "
          f"(the CLI cannot PUT this in one request)", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp, open(local, "rb") as handle:
        for i in range(n_parts):
            buf = handle.read(chunk)
            digest.update(buf)
            part = Path(tmp) / f"part{i:04d}"
            part.write_bytes(buf)
            try:
                _upload_one(part, f"{remote}.part{i:04d}")
            finally:
                part.unlink(missing_ok=True)
            set_progress("uploading the lens to the new VM",
                         f"chunk {i + 1}/{n_parts}")

    # Idempotent on purpose. `exec --timeout` bounds only the CLIENT wait -- if the
    # socket stalls, the kernel finishes the join anyway and deletes the parts. A
    # naive retry would then find no parts and truncate the file to nothing, so the
    # no-parts-but-file-exists case re-hashes what is already there instead.
    joiner = f"""
import glob, hashlib, os
remote = {remote!r}
def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

parts = sorted(glob.glob(remote + ".part*"))
if not parts and os.path.exists(remote):
    print("JOINED", 0, os.path.getsize(remote), digest(remote))
else:
    with open(remote, "wb") as out:
        for p in parts:
            with open(p, "rb") as fh:
                while True:
                    b = fh.read(1 << 20)
                    if not b:
                        break
                    out.write(b)
    for p in parts:
        os.remove(p)
    print("JOINED", len(parts), os.path.getsize(remote), digest(remote))
"""
    out = exec_code(joiner, timeout=600, retries=1)
    line = next((l for l in out.splitlines() if l.startswith("JOINED")), "")
    fields = line.split()
    if len(fields) != 4:
        raise RuntimeError(f"reassembly failed on the VM: {out.strip()[-500:]}")
    _, n_seen, remote_size, remote_sha = fields
    if int(remote_size) != size or remote_sha != digest.hexdigest():
        raise RuntimeError(
            f"upload corrupted: local {size}b/{digest.hexdigest()[:12]} vs "
            f"remote {remote_size}b/{remote_sha[:12]} from {n_seen} parts")
    print(f"[panel] upload verified ({remote_size} bytes, sha256 matches)",
          file=sys.stderr)


def exec_code(code: str, timeout: int | None = None, retries: int = 0) -> str:
    """Run `code` in the persistent kernel and return stdout.

    The exec websocket occasionally stalls (this project has seen
    `WebSocketConnectionClosedException` mid-call). Because `--timeout` is only a
    client-side wait, a stall used to block for the full budget and then surface a
    bare `Command ... timed out` with nothing actionable. Keep per-operation budgets
    tight and retry, so a stalled socket costs one short wait instead of many
    minutes. Only pass retries>0 for work that is safe to run twice.
    """
    t = timeout or Config.timeout
    for attempt in range(retries + 1):
        try:
            with _colab_lock:
                proc = colab("exec", "-s", Config.session, "--timeout", str(t),
                             stdin=code, timeout=t + 60)
        except subprocess.TimeoutExpired:
            print(f"[panel] exec stalled past {t + 60}s "
                  f"(attempt {attempt + 1}/{retries + 1}); the kernel may still be "
                  f"running it", file=sys.stderr)
            if attempt == retries:
                raise RuntimeError(
                    f"the Colab exec connection stalled ({t + 60}s). The kernel is "
                    f"probably fine — press Run again. If it keeps happening, "
                    f"restart the kernel: colab --auth=adc restart-kernel -s "
                    f"{Config.session}") from None
            continue
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip()
                               or "exec failed")
        return proc.stdout
    raise RuntimeError("unreachable")


def extract_payload(stdout: str, dest: Path) -> dict:
    """Pull the fenced base64 tarball out of Colab's noisy stdout into `dest`."""
    start = stdout.find(BEGIN)
    end = stdout.find(END)
    if start < 0 or end < 0:
        tail = stdout.strip()[-1500:]
        raise RuntimeError(f"no payload in kernel output; tail:\n{tail}")
    blob = stdout[start + len(BEGIN):end].strip()
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(blob)), mode="r:gz") as tar:
        tar.extractall(dest)          # trusted: we produced this tarball ourselves
    meta = dest / "panel.json"
    return json.loads(meta.read_text()) if meta.exists() else {}


def vis_page() -> str:
    """jlens's fetch-mode slice page, downloaded and cached once."""
    cache = Config.root / ".cache" / "slice_vis.html"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SLICE_VIS_URL, timeout=60) as r:
            cache.write_bytes(r.read())
    return (cache.read_text(encoding="utf-8")
            .replace("__D3__", D3_TAG)
            .replace("__TITLE__", "Gemma 4 12B QAT")
            .replace("__WHAT__", "Jacobian lens readout")
            .replace("__BOOTSTRAP__", '{"mode":"fetch"}'))


SHELL = r"""<!doctype html>
<meta charset="utf-8"><title>Gemma 4 lens panel</title>
<style>
 :root{--bg:#fbfbfa;--fg:#1d1c1a;--mut:#6b6862;--line:#dedbd5;--acc:#2f6f4e;--card:#fff}
 @media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#eeece7;--mut:#9a958c;
   --line:#33312d;--acc:#7fc4a0;--card:#1c1b19}}
 *{box-sizing:border-box}
 body{margin:0;font:14px/1.5 ui-sans-serif,system-ui,sans-serif;background:var(--bg);color:var(--fg)}
 header{display:flex;align-items:center;gap:12px;padding:10px 16px;border-bottom:1px solid var(--line)}
 header h1{font-size:15px;margin:0;font-weight:600;letter-spacing:-.01em}
 .grow{flex:1}
 .wrap{display:grid;grid-template-columns:var(--sidew,420px) 1fr;height:calc(100vh - 49px)}
 .side{border-right:1px solid var(--line);display:flex;flex-direction:column;
   overflow:hidden;min-height:0}
 .sidescroll{flex:1 1 auto;overflow:auto;padding:14px;display:flex;
   flex-direction:column;gap:12px;min-height:0}
 /* Run + status pinned: the status line carries the "did it work" answer, and it
    was rendering below the fold at the bottom of a scrolling sidebar. */
 .sidefoot{flex:0 0 auto;border-top:1px solid var(--line);padding:12px 14px;
   background:var(--card);display:flex;flex-direction:column;gap:8px}
 #grip{position:absolute;top:49px;bottom:0;width:6px;cursor:col-resize;z-index:5}
 #grip:hover{background:var(--acc);opacity:.35}
 label{display:block;font-size:12px;color:var(--mut);margin-bottom:4px}
 textarea,input[type=number],select{width:100%;padding:7px 8px;border:1px solid var(--line);
   border-radius:6px;background:var(--card);color:var(--fg);font:inherit}
 textarea{min-height:96px;resize:vertical;font-family:ui-monospace,monospace;font-size:12.5px}
 .side textarea{line-height:1.45}
 .chk{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--fg)}
 .chk input{width:auto;margin:0}
 button{padding:7px 12px;border:1px solid var(--line);border-radius:6px;background:var(--card);
   color:var(--fg);font:inherit;cursor:pointer}
 button.primary{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600}
 button:disabled{opacity:.5;cursor:not-allowed}
 button.danger{border-color:#b4453c;color:#b4453c}
 .row{display:flex;gap:8px;align-items:center}
 .row>*{flex:1}
 #drop{border:1.5px dashed var(--line);border-radius:8px;padding:14px;text-align:center;
   color:var(--mut);font-size:12px;cursor:pointer}
 #drop.on{border-color:var(--acc);color:var(--acc)}
 #thumb{max-width:100%;border-radius:6px;margin-top:8px;display:none}
 #status{font-size:12px;color:var(--mut);min-height:18px;white-space:pre-wrap;
   max-height:120px;overflow:auto}
 #answer{font-family:ui-monospace,monospace;font-size:12px;background:var(--bg);
   border:1px solid var(--line);border-radius:6px;padding:8px;max-height:150px;
   overflow:auto;white-space:pre-wrap;color:var(--fg)}
 .sidefoot{max-height:52vh;overflow:auto}
 #viewwrap{position:relative;min-width:0;overflow:hidden}
 iframe{width:100%;height:100%;border:0;background:var(--card)}
 #placeholder{height:100%;display:flex;align-items:center;justify-content:center;
   text-align:center;color:var(--mut);padding:32px}
 #placeholder>div{max-width:460px}
 #placeholder p{margin:10px 0 0}
 fieldset{border:1px solid var(--line);border-radius:8px;padding:10px;margin:0}
 legend{font-size:11px;color:var(--mut);padding:0 4px}
 #steps{display:none}
 .hint{font-size:11px;color:var(--mut)}
</style>
<header>
  <h1>Gemma 4 12B QAT &mdash; Jacobian lens</h1>
  <span class="grow"></span>
  <span id="sess" class="hint"></span>
  <button id="stop" class="danger">Stop Colab session</button>
</header>
<div class="wrap">
 <div class="side">
  <div class="sidescroll">
  <div>
    <label>System prompt</label>
    <textarea id="system" style="min-height:84px"
      placeholder="(optional) e.g. You are a concise research assistant."></textarea>
  </div>
  <div>
    <label>Conversation</label>
    <div id="msgs"></div>
    <div class="row" style="margin-top:6px">
      <button id="adduser">+ user turn</button>
      <button id="addasst">+ assistant turn</button>
    </div>
  </div>
  <div>
    <label>Assistant prefill <span class="hint">(readout mid-answer)</span></label>
    <textarea id="prefill" style="min-height:66px"
      placeholder="(optional) text the assistant has already begun"></textarea>
  </div>
  <div>
    <label>Image (optional &mdash; Gemma 4 reads images)</label>
    <div id="drop">drop an image, or click to choose</div>
    <input id="file" type="file" accept="image/*" hidden>
    <img id="thumb">
    <div class="row" style="margin-top:6px"><button id="clearimg"
      style="display:none">remove image</button></div>
  </div>
  <fieldset>
    <legend>view</legend>
    <div class="row">
      <div><label>mode</label>
        <select id="mode">
          <option value="slice">prompt slice</option>
          <option value="generation">prompt + generation</option>
        </select></div>
      <div id="steps"><label>max tokens</label>
        <input id="nsteps" type="number" value="64" min="1" max="256"></div>
    </div>
    <div class="row" style="margin-top:8px">
      <div><label>formatting</label>
        <select id="chat">
          <option value="1">chat template (recommended)</option>
          <option value="0">raw text</option>
        </select></div>
    </div>
    <div class="row" style="margin-top:8px">
      <div><label>top-k</label><input id="topn" type="number" value="10" min="1" max="25"></div>
      <div><label>layer stride</label><input id="stride" type="number" value="1" min="1" max="8"></div>
    </div>
    <div class="row" style="margin-top:8px">
      <div><label>last N tokens</label><input id="lastn" type="number" value="0" min="0"
        placeholder="0 = all"></div>
      <div><label>max seq len</label><input id="maxlen" type="number" value="512" min="32"></div>
    </div>
    <label class="chk" style="margin-top:10px">
      <input id="mask" type="checkbox" checked>
      word-like tokens only
    </label>
    <div class="hint">Hides <code>&lt;image|&gt;</code>/<code>&lt;audio|&gt;</code> and
      fragment tokens from the displayed top-k. Ranks stay full-vocab. Worth leaving on
      until the lens is fitted on more prompts.</div>
  </fieldset>
  <div class="hint">Compute runs on the Colab A100; only the rendered view is local.
    The first run boots the model into the kernel and takes a couple of minutes.</div>
  </div><!-- /sidescroll -->
  <div class="sidefoot">
    <button id="run" class="primary">Run lens</button>
    <div id="status"></div>
    <div id="stepbar" style="display:none">
      <label>answer <span class="hint">(what the model generated)</span></label>
      <div id="answer"></div>
      <label style="margin-top:8px">slice at step <span id="steplabel">0</span>
        of <span id="stepmax">0</span></label>
      <input id="stepslider" type="range" min="0" max="0" value="0" style="width:100%">
      <div id="steptext" class="hint"></div>
    </div>
  </div>
 </div>
 <div id="grip"></div>
 <div id="viewwrap">
   <div id="placeholder">
     <div>
       <strong>No readout yet.</strong>
       <p>Set up a conversation on the left and press <em>Run lens</em>.<br>
          Rows are layers, columns are token positions; each cell shows what that
          layer is disposed to say at that position.</p>
       <p class="hint">The current lens is fitted on 3 prompts, so only the last
          layers (~L40+) are meaningful — early rows are noise until it is fitted
          on more.</p>
     </div>
   </div>
   <iframe id="view" src="about:blank" style="display:none"></iframe>
 </div>
</div>
<script>
// ---- draggable sidebar divider ------------------------------------------
(function () {
  const grip = document.getElementById('grip'), wrap = document.querySelector('.wrap');
  const place = () => {
    const w = parseInt(getComputedStyle(document.documentElement)
      .getPropertyValue('--sidew')) || 420;
    grip.style.left = (w - 3) + 'px';
  };
  let drag = false;
  grip.onmousedown = e => { drag = true; e.preventDefault(); };
  window.onmousemove = e => {
    if (!drag) return;
    const w = Math.min(Math.max(e.clientX, 280), window.innerWidth - 320);
    document.documentElement.style.setProperty('--sidew', w + 'px');
    place();
  };
  window.onmouseup = () => { drag = false; };
  window.addEventListener('resize', place);
  place();
})();
</script>
<script>
const $ = s => document.querySelector(s);
let imageB64 = null, gen = null;

// ---- conversation editor -------------------------------------------------
let msgs = [{role: 'user',
  content: 'Fact: The currency used in the country shaped like a boot is'}];

function renderMsgs() {
  const host = $('#msgs');
  host.innerHTML = '';
  msgs.forEach((m, i) => {
    const box = document.createElement('div');
    box.style.cssText = 'border:1px solid var(--line);border-radius:8px;padding:8px;'
      + 'margin-bottom:6px;background:var(--card)';
    const bar = document.createElement('div');
    bar.className = 'row';
    bar.style.marginBottom = '4px';
    const sel = document.createElement('select');
    ['user', 'assistant'].forEach(r => {
      const o = document.createElement('option');
      o.value = r; o.textContent = r; o.selected = m.role === r; sel.appendChild(o);
    });
    sel.onchange = e => { msgs[i].role = e.target.value; };
    const del = document.createElement('button');
    del.textContent = 'remove';
    del.style.flex = '0 0 auto';
    del.onclick = () => { msgs.splice(i, 1); if (!msgs.length)
      msgs.push({role: 'user', content: ''}); renderMsgs(); };
    bar.append(sel, del);
    const ta = document.createElement('textarea');
    ta.value = m.content;
    ta.style.minHeight = '92px';
    ta.oninput = e => { msgs[i].content = e.target.value; };
    box.append(bar, ta);
    host.appendChild(box);
  });
}
$('#adduser').onclick = () => { msgs.push({role: 'user', content: ''}); renderMsgs(); };
$('#addasst').onclick = () => { msgs.push({role: 'assistant', content: ''}); renderMsgs(); };
renderMsgs();

$('#mode').onchange = () => {
  const g = $('#mode').value === 'generation';
  // must be an explicit value: '' only clears the inline style, and the #steps
  // CSS rule below would keep it display:none.
  $('#steps').style.display = g ? 'block' : 'none';
};
$('#drop').onclick = () => $('#file').click();
$('#file').onchange = e => e.target.files[0] && loadImage(e.target.files[0]);
$('#drop').ondragover = e => { e.preventDefault(); $('#drop').classList.add('on'); };
$('#drop').ondragleave = () => $('#drop').classList.remove('on');
$('#drop').ondrop = e => {
  e.preventDefault(); $('#drop').classList.remove('on');
  const f = e.dataTransfer.files[0]; if (f) loadImage(f);
};
$('#clearimg').onclick = () => {
  imageB64 = null; $('#thumb').style.display = 'none'; $('#file').value = '';
  $('#drop').textContent = 'drop an image, or click to choose';
  $('#clearimg').style.display = 'none';
};
function loadImage(file) {
  const r = new FileReader();
  r.onload = () => {
    imageB64 = r.result.split(',')[1];
    $('#thumb').src = r.result; $('#thumb').style.display = 'block';
    $('#drop').textContent = file.name;
    $('#clearimg').style.display = '';
  };
  r.readAsDataURL(file);
}

function setStatus(t) { $('#status').textContent = t; }

function showView(src) {
  $('#placeholder').style.display = 'none';
  $('#view').style.display = '';
  $('#view').src = src;
}

$('#run').onclick = async () => {
  const body = {
    messages: msgs,
    system: $('#system').value,
    assistant_prefill: $('#prefill').value,
    chat: $('#chat').value === '1',
    image_b64: imageB64,
    mode: $('#mode').value,
    max_new_tokens: +$('#nsteps').value,
    top_n: +$('#topn').value,
    layer_stride: +$('#stride').value,
    last_n_tokens: +$('#lastn').value || null,
    max_seq_len: +$('#maxlen').value,
    mask_display: $('#mask').checked,
  };
  $('#run').disabled = true;
  const t0 = Date.now();
  // Ask the server what it is actually doing. A cold start (fresh VM -> lens
  // upload -> 10GB weight download) takes minutes, and reporting it as
  // "computing slice" made it look like a hang.
  let phase = 'starting', detail = '';
  const tick = setInterval(async () => {
    const s = ((Date.now() - t0) / 1000).toFixed(0);
    try {
      const p = await (await fetch('/api/progress')).json();
      if (p.phase && p.phase !== 'idle') { phase = p.phase; detail = p.detail || ''; }
    } catch (e) { /* keep the last known phase */ }
    setStatus(`${phase}${detail ? ' — ' + detail : ''} · ${s}s`
      + (s > 90 ? '\n(first run on a fresh VM takes ~6 min: lens upload + model load)'
                : ''));
  }, 700);
  $('#stepbar').style.display = 'none';
  try {
    const r = await fetch('/api/slice', {method: 'POST',
      headers: {'content-type': 'application/json'}, body: JSON.stringify(body)});
    const j = await r.json();
    clearInterval(tick);        // stop before writing the final status, or it overwrites
    if (!r.ok) throw new Error(j.error || 'failed');
    if (j.panel.kind === 'generation') {
      gen = j;
      const sl = j.panel.slices || [];
      $('#answer').textContent = j.panel.final_text || '(nothing generated)';
      $('#stepslider').max = Math.max(0, sl.length - 1);
      $('#stepslider').value = Math.max(0, sl.length - 1);   // land on the full answer
      $('#stepmax').textContent = j.panel.steps.length;
      $('#stepbar').style.display = '';
      showStep(sl.length - 1);
      setStatus(`${j.panel.steps.length} tokens generated · ${sl.length} slices`
        + ` (every ${j.panel.slice_every})`);
    } else {
      showView(`/vis.html?datapath=/data/${j.run_id}/`);
      const top = (j.panel.final_top || []).map(t => JSON.stringify(t)).join('  ');
      setStatus(`seq_len ${j.panel.seq_len} · ${j.panel.layers.length} layers`
        + ` · ${j.panel.n_tracked} tracked` + (j.panel.has_image ? ' · with image' : '')
        + (top ? `\nmodel predicts next: ${top}` : ''));
    }
  } catch (e) { setStatus('error: ' + e.message); }
  clearInterval(tick);
  $('#run').disabled = false;
};

function showStep(i) {
  const sl = gen.panel.slices || [];
  const step = sl[Math.max(0, Math.min(i, sl.length - 1))];
  if (step === undefined) return;
  $('#steplabel').textContent = step;
  const upto = gen.panel.steps.slice(0, step).map(s => s.token).join('');
  $('#steptext').textContent = upto ? `generated so far: ${JSON.stringify(upto)}`
                                    : '(prompt only, before any generation)';
  showView(`/vis.html?datapath=/data/${gen.run_id}/step${String(step).padStart(3,'0')}/`);
}
$('#stepslider').oninput = e => gen && showStep(+e.target.value);

$('#stop').onclick = async () => {
  if (!confirm('Stop the Colab session? The VM is released and the model unloads.')) return;
  setStatus('stopping session...');
  const r = await fetch('/api/stop', {method: 'POST'});
  const j = await r.json();
  setStatus(j.message || 'stopped');
  $('#sess').textContent = 'stopped';
};

fetch('/api/status').then(r => r.json())
  .then(j => $('#sess').textContent = j.summary || '')
  .catch(() => {});
</script>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter console
        if "/api/" in (args[0] if args else ""):
            sys.stderr.write("  %s\n" % (fmt % args))

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            return self._send(200, SHELL.encode(), "text/html; charset=utf-8")
        if path == "/vis.html":
            return self._send(200, vis_page().encode(), "text/html; charset=utf-8")
        if path == "/api/progress":
            return self._json(200, dict(_progress))
        if path == "/favicon.ico":
            return self._send(200, b"", "image/x-icon")   # silence the browser's 404
        if path == "/api/status":
            proc = colab("status", "-s", Config.session, timeout=120)
            return self._json(200, {"summary": proc.stdout.strip().splitlines()[0]
                                    if proc.stdout.strip() else "no session"})
        if path.startswith("/data/"):
            rel = path[len("/data/"):]
            target = (Config.root / "runs" / rel).resolve()
            if not str(target).startswith(str((Config.root / "runs").resolve())):
                return self._json(403, {"error": "path escape"})
            if not target.is_file():
                return self._json(404, {"error": "not found"})
            ctype = ("application/json" if target.suffix == ".json"
                     else "application/octet-stream")
            return self._send(200, target.read_bytes(), ctype)
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")

        if path == "/api/stop":
            proc = colab("stop", "-s", Config.session, timeout=180)
            return self._json(200, {"message": (proc.stdout or proc.stderr).strip()})

        if path == "/api/slice":
            try:
                return self._json(200, run_slice(body))
            except Exception as exc:  # surfaced in the panel's status line
                return self._json(500, {"error": str(exc)})

        self._json(404, {"error": "not found"})


def run_slice(req: dict) -> dict:
    """Boot the service if needed, compute a slice, unpack it under panel/runs/."""
    ensure_booted()
    opts = {
        "top_n": int(req.get("top_n", 10)),
        "layer_stride": int(req.get("layer_stride", 1)),
        "last_n_tokens": req.get("last_n_tokens") or None,
        "max_seq_len": int(req.get("max_seq_len", 512)),
        "mask_display": bool(req.get("mask_display", True)),
        "messages": req.get("messages") or None,
        "system": (req.get("system") or "").strip() or None,
        "assistant_prefill": req.get("assistant_prefill") or "",
        "chat": bool(req.get("chat", True)),
    }
    prompt = req.get("prompt")
    image_b64 = req.get("image_b64")

    if req.get("mode") == "generation":
        call = (f"print(generation_payload({prompt!r}, image_b64={image_b64!r}, "
                f"max_new_tokens={int(req.get('max_new_tokens', 8))}, **{opts!r}))")
        # ~1s/step of GPU work plus payload transfer; 8 steps measured at 8.1s.
        timeout = min(Config.timeout, 120 + 30 * int(req.get("max_new_tokens", 8)))
    else:
        call = f"print(slice_payload({prompt!r}, image_b64={image_b64!r}, **{opts!r}))"
        timeout = min(Config.timeout, 240)   # measured ~1s text, ~2s image

    run_id = uuid.uuid4().hex[:12]
    dest = Config.root / "runs" / run_id
    # Safe to retry: recomputing a slice has no side effects on the kernel.
    set_progress("computing on the A100",
                 "generation: one slice per step" if req.get("mode") == "generation"
                 else "slice")
    try:
        panel = extract_payload(exec_code(call, timeout=timeout, retries=1), dest)
    finally:
        set_progress("idle")
    return {"run_id": run_id, "panel": panel}


_booted = threading.Event()

REMOTE_LENS = "/content/gemma4_12b_qat_lens.pt"


def ensure_session() -> None:
    """Create the Colab session if it isn't already up.

    Lets the panel be a single command: it provisions the GPU on the first
    request rather than making you run `colab new` separately. Colab also
    reclaims idle VMs, so this doubles as recovery after a pruned session.
    """
    proc = colab("sessions", timeout=180)
    if f"[{Config.session}]" in (proc.stdout or ""):
        return
    hardware = ["--gpu", Config.gpu] if Config.gpu.lower() not in ("", "cpu") else []
    set_progress("provisioning a fresh Colab VM",
                 "the previous one was reclaimed; this takes ~30s")
    print(f"[panel] no session '{Config.session}'; provisioning "
          f"{Config.gpu or 'CPU'}...", file=sys.stderr)
    new = colab("new", "-s", Config.session, *hardware, timeout=900)
    if new.returncode != 0:
        raise RuntimeError(f"could not create session: "
                           f"{(new.stderr or new.stdout).strip()[-600:]}")
    inst = colab("install", "-s", Config.session, "compressed-tensors", "accelerate",
                 "hf_transfer", timeout=900)
    if inst.returncode != 0:
        print(f"[panel] warning: install step failed, the service will retry: "
              f"{(inst.stderr or inst.stdout).strip()[-300:]}", file=sys.stderr)


def fetch_lens_from_hub() -> None:
    """Have the VM pull the lens from HF Hub, rather than pushing it from here.

    The push is bounded by *your* uplink -- measured 4.7 MB/s, so ~170s for 649MB,
    and paid again every time Colab reclaims the VM. Colab downloading from the Hub
    runs at datacentre bandwidth instead, and needs no local copy at all.
    """
    code = f'''
import hashlib, os, shutil, sys, time
try:
    import hf_transfer  # noqa: F401  -- multipart download; the point of doing this
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
except ImportError:
    pass
from huggingface_hub import hf_hub_download

t0 = time.time()
path = hf_hub_download({Config.hf_repo!r}, filename={Config.hf_file!r},
                       local_dir="/content/_lens_dl")
dt = time.time() - t0

digest = hashlib.sha256()
with open(path, "rb") as fh:
    for block in iter(lambda: fh.read(1 << 22), b""):
        digest.update(block)
got, want = digest.hexdigest(), {Config.hf_sha256!r}
if want and got != want:
    print("HUB_BAD_SHA got=" + got)
    sys.exit(0)

os.replace(path, {REMOTE_LENS!r})          # same filesystem, so this is a rename
shutil.rmtree("/content/_lens_dl", ignore_errors=True)
mb = os.path.getsize({REMOTE_LENS!r}) / 1e6
print(f"HUB_OK {{mb:.0f}}MB in {{dt:.0f}}s ({{mb / max(dt, 0.1):.0f}} MB/s)")
'''
    proc = colab("exec", "-s", Config.session, "--timeout", "600", stdin=code,
                 timeout=720)
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if "HUB_OK" not in out:
        raise RuntimeError(out[-500:] or "no output from the hub download")
    print(f"[panel] {out[out.index('HUB_OK'):].splitlines()[0]}", file=sys.stderr)


def push_lens_from_disk() -> None:
    """Fallback: chunked upload of the local lens. See upload_large()."""
    if not Config.lens.exists():
        raise RuntimeError(
            f"no lens on the Hub and none locally at {Config.lens}; fit one with "
            f"scripts/fit_gemma4_lens.py and download it there")
    size_mb = Config.lens.stat().st_size / 1e6
    set_progress("uploading the lens from this machine",
                 f"{size_mb:.0f} MB in 32 MB chunks; ~3 min on a home uplink")
    print(f"[panel] uploading lens ({size_mb:.0f} MB) to the VM...", file=sys.stderr)
    try:
        upload_large(Config.lens, REMOTE_LENS)
    except Exception as exc:
        raise RuntimeError(f"lens upload failed: {exc}") from exc


def ensure_booted() -> None:
    """Make the session usable: get the lens onto the VM, then load the service.

    Colab reclaims idle VMs (a 404/401 on exec means the backend pruned it), which
    takes /content with it -- so the lens is fetched rather than assumed present.
    """
    if _booted.is_set():
        return
    with _colab_lock:
        if _booted.is_set():
            return
        ensure_session()

        probe = ("import os; print('LENS_PRESENT' if os.path.exists("
                 f"{REMOTE_LENS!r}) else 'LENS_MISSING')")
        proc = colab("exec", "-s", Config.session, "--timeout", "60", stdin=probe,
                     timeout=180)
        if "LENS_PRESENT" not in proc.stdout:
            if Config.hf_repo:
                set_progress("fetching the lens from HF Hub",
                             f"{Config.hf_repo}, at Colab's bandwidth")
                try:
                    fetch_lens_from_hub()
                except Exception as exc:
                    print(f"[panel] hub fetch failed ({exc});\n[panel] falling back "
                          f"to uploading from this machine", file=sys.stderr)
                    push_lens_from_disk()
            else:
                push_lens_from_disk()

        set_progress("loading Gemma 4 into the kernel",
                     "downloading ~10GB of weights and decompressing; ~3 min")
        proc = colab("exec", "-s", Config.session, "--timeout", "1800",
                     "-f", str(Config.service), timeout=1920)
        if proc.returncode != 0:
            raise RuntimeError(f"service boot failed: "
                               f"{(proc.stderr or proc.stdout).strip()[-1200:]}")
        _booted.set()
        print("[panel] service ready", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="lens", help="Colab session name")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--timeout", type=int, default=900,
                    help="per-exec timeout in seconds")
    ap.add_argument("--fresh", action="store_true", help="clear cached runs first")
    ap.add_argument("--lens", type=Path, default=None,
                    help=f"local lens .pt, used only as an HF fallback "
                         f"(default: {Config.lens})")
    ap.add_argument("--hf-repo", default=Config.hf_repo,
                    help=f"HF repo the VM pulls the lens from "
                         f"(default: {Config.hf_repo})")
    ap.add_argument("--no-hf", action="store_true",
                    help="skip the Hub and upload the local lens instead")
    ap.add_argument("--gpu", default="A100",
                    help="GPU to provision if the session doesn't exist "
                         "(needs ~30GB: A100 or H100)")
    args = ap.parse_args()

    Config.session = args.session
    Config.timeout = args.timeout
    Config.gpu = args.gpu
    Config.hf_repo = "" if args.no_hf else args.hf_repo
    if Config.hf_repo != ap.get_default("hf_repo"):
        Config.hf_sha256 = ""      # a different repo won't match the pinned hash
    if args.lens:
        Config.lens = args.lens.resolve()
    if args.fresh:
        shutil.rmtree(Config.root / "runs", ignore_errors=True)
    (Config.root / "runs").mkdir(parents=True, exist_ok=True)

    if not shutil.which("colab"):
        print("colab CLI not found on PATH", file=sys.stderr)
        return 2

    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        if exc.errno != 98:
            raise
        print(f"port {args.port} is already in use — an older panel is probably still "
              f"running.\nStop it (pkill -f '[l]ens_pane') or pass --port "
              f"{args.port + 1}.\nNote a stale panel serves stale code: restart it after "
              f"pulling changes.", file=sys.stderr)
        return 2
    print(f"[panel] http://127.0.0.1:{args.port}  (session '{Config.session}')")
    print("[panel] ctrl-c to stop the panel; use the button in the UI to stop the VM")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[panel] bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
