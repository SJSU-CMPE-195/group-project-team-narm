// Browser-side collector UI for the Colab ASL data collection notebooks.
//
// This file is inlined by make_notebooks.py into the "Run collection" cell and
// executed with IPython.display.Javascript. The config placeholder on the
// "const CFG" line below is replaced with a JSON object built by
// helpers.py:ui_config(). (The placeholder must appear exactly once in this file.)
//
// Flow per take:
//   flash word -> countdown 5..1 -> record N frames (corner counter) ->
//   upload frames to Python -> MediaPipe QA -> review/replay -> Good (save) / Bad (retry)
//
// Frames are always saved UNMIRRORED. MIRROR_PREVIEW only changes the live preview.
(async () => {
  const CFG = __CFG__;
  const FPS = CFG.fps;
  const N = CFG.numFrames;
  const W = CFG.width;
  const H = CFG.height;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const S = { stop: false };
  const ui = { mode: 'live', banner: null, count: null, rec: null, slate: null };
  let replay = null;
  let conn = { pose: [], hand: [], faceStride: 3 };

  // ---------------------------------------------------------------- DOM
  const root = document.createElement('div');
  root.style.cssText =
    'font-family:system-ui,Segoe UI,Roboto,sans-serif;max-width:780px;background:#16181d;' +
    'color:#e8eaed;padding:12px;border-radius:10px;box-sizing:border-box;';
  root.innerHTML = `
    <style>
      .aslb{padding:8px 14px;border:0;border-radius:6px;font-size:14px;cursor:pointer;background:#3a3f4b;color:#fff}
      .aslb:hover{filter:brightness(1.15)}
      .aslb.good{background:#1e8e3e;font-weight:600}
      .aslb.bad{background:#d93025;font-weight:600}
      .aslb.on{outline:2px solid #8ab4f8}
      .aslb:disabled{opacity:.5;cursor:default}
      .asl-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
    </style>
    <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;">
      <div id="hdrWord" style="font-size:20px;font-weight:700;"></div>
      <div id="hdrCount" style="font-size:14px;opacity:.85;"></div>
    </div>
    <canvas id="view" width="${W}" height="${H}"
            style="width:100%;background:#000;border-radius:6px;display:block;"></canvas>
    <div id="msg" style="min-height:22px;margin:8px 0;font-size:14px;"></div>
    <div id="live" class="asl-row"></div>
    <div id="review" style="display:none;">
      <div id="qa" style="font-size:13px;line-height:1.5;margin-bottom:8px;"></div>
      <div class="asl-row">
        <button class="aslb" id="bPlay">Pause</button>
        <button class="aslb" id="bRestart">Replay</button>
        <span style="font-size:13px;">Speed:</span>
        <button class="aslb spd" data-s="1">1x</button>
        <button class="aslb spd" data-s="0.5">0.5x</button>
        <button class="aslb spd" data-s="0.25">0.25x</button>
        <label style="font-size:13px;"><input type="checkbox" id="cSkel"> Show skeleton</label>
      </div>
      <input type="range" id="scrub" min="0" max="${N - 1}" value="0" style="width:100%;">
      <div class="asl-row" style="margin-top:10px;">
        <button class="aslb good" id="bGood">Good - save</button>
        <button class="aslb bad" id="bBad">Bad - retry</button>
        <button class="aslb" id="bStop">Stop session</button>
      </div>
    </div>`;
  document.body.appendChild(root);
  const $ = (id) => root.querySelector('#' + id);
  const view = $('view');
  const vctx = view.getContext('2d');

  const say = (text, color) => {
    $('msg').textContent = text;
    $('msg').style.color = color || '#e8eaed';
  };
  const fail = (err) => {
    const m = err && err.message ? err.message : String(err);
    say('Error: ' + m + '  (re-run this cell to restart; saved takes are safe)', '#f28b82');
    $('live').innerHTML = '';
    $('review').style.display = 'none';
  };

  // ------------------------------------------------- Python bridge
  async function call(name, args) {
    if (!(window.google && google.colab && google.colab.kernel)) {
      throw new Error('google.colab.kernel not found - this notebook must run inside Google Colab.');
    }
    const r = await google.colab.kernel.invokeFunction(name, args || [], {});
    const out = r && r.data && r.data['application/json'];
    if (!out) throw new Error('No response from Python for ' + name);
    if (out.error) throw new Error(out.error);
    return out;
  }

  // ------------------------------------------------------ camera
  let stream = null;
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  const cap = document.createElement('canvas');
  cap.width = W;
  cap.height = H;
  const cctx = cap.getContext('2d');
  let presented = -1;
  let presentedSupported = false;
  let camFps = null;

  // Draw the current camera frame into the W x H capture canvas, center-cropped
  // to the target aspect ratio (never stretched) so geometry stays undistorted.
  function drawCap() {
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return;
    const targetAR = W / H;
    const srcAR = vw / vh;
    let sx = 0, sy = 0, sw = vw, sh = vh;
    if (srcAR > targetAR) {
      sw = vh * targetAR;
      sx = (vw - sw) / 2;
    } else {
      sh = vw / targetAR;
      sy = (vh - sh) / 2;
    }
    cctx.drawImage(video, sx, sy, sw, sh, 0, 0, W, H);
  }

  async function startCamera() {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    for (let i = 0; i < 100 && !video.videoWidth; i++) await sleep(50);
    if (!video.videoWidth) throw new Error('Camera started but returned no video frames.');
    const track = stream.getVideoTracks()[0];
    const st = track && track.getSettings ? track.getSettings() : {};
    camFps = st.frameRate || null;
    if ('requestVideoFrameCallback' in video) {
      presentedSupported = true;
      const cb = (_now, md) => {
        presented = md.presentedFrames;
        video.requestVideoFrameCallback(cb);
      };
      video.requestVideoFrameCallback(cb);
    }
  }

  function stopCamera() {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }

  // ----------------------------------------------------- drawing
  function fitFont(ctx, text, maxW, startPx, weight) {
    let px = startPx;
    do {
      ctx.font = `${weight} ${px}px system-ui,Segoe UI,Roboto,sans-serif`;
      if (ctx.measureText(text).width <= maxW) break;
      px -= 4;
    } while (px > 16);
    return px;
  }

  function centerText(text, y, startPx, color, weight) {
    fitFont(vctx, text, W * 0.9, startPx, weight || 'bold');
    vctx.fillStyle = color;
    vctx.textAlign = 'center';
    vctx.textBaseline = 'middle';
    vctx.fillText(text, W / 2, y);
  }

  function drawSkeleton(ctx, ov) {
    const SC = 1000;
    ctx.lineWidth = 2;
    const P = ov.p || [];
    if (P.length) {
      ctx.strokeStyle = 'rgba(138,180,248,0.9)';
      for (const [a, b] of conn.pose) {
        if (a * 3 + 2 >= P.length || b * 3 + 2 >= P.length) continue;
        if (P[a * 3 + 2] < 500 || P[b * 3 + 2] < 500) continue;
        ctx.beginPath();
        ctx.moveTo((P[a * 3] / SC) * W, (P[a * 3 + 1] / SC) * H);
        ctx.lineTo((P[b * 3] / SC) * W, (P[b * 3 + 1] / SC) * H);
        ctx.stroke();
      }
      ctx.fillStyle = 'rgba(232,234,237,0.95)';
      for (let i = 0; i + 2 < P.length; i += 3) {
        if (P[i + 2] < 500) continue;
        ctx.beginPath();
        ctx.arc((P[i] / SC) * W, (P[i + 1] / SC) * H, 3, 0, 6.2832);
        ctx.fill();
      }
    }
    const F = ov.f || [];
    ctx.fillStyle = 'rgba(160,160,160,0.8)';
    for (let i = 0; i + 1 < F.length; i += 2) {
      ctx.fillRect((F[i] / SC) * W - 1, (F[i + 1] / SC) * H - 1, 2, 2);
    }
    const hand = (arr, line, dot) => {
      if (!arr || !arr.length) return;
      ctx.strokeStyle = line;
      for (const [a, b] of conn.hand) {
        if (a * 2 + 1 >= arr.length || b * 2 + 1 >= arr.length) continue;
        ctx.beginPath();
        ctx.moveTo((arr[a * 2] / SC) * W, (arr[a * 2 + 1] / SC) * H);
        ctx.lineTo((arr[b * 2] / SC) * W, (arr[b * 2 + 1] / SC) * H);
        ctx.stroke();
      }
      ctx.fillStyle = dot;
      for (let i = 0; i + 1 < arr.length; i += 2) {
        ctx.beginPath();
        ctx.arc((arr[i] / SC) * W, (arr[i + 1] / SC) * H, 3, 0, 6.2832);
        ctx.fill();
      }
    };
    hand(ov.l, 'rgba(129,201,149,0.95)', 'rgba(129,201,149,1)'); // left hand: green
    hand(ov.r, 'rgba(253,214,99,0.95)', 'rgba(253,214,99,1)'); // right hand: yellow
  }

  let lastT = performance.now();
  function render() {
    requestAnimationFrame(render);
    const now = performance.now();
    const dt = now - lastT;
    lastT = now;

    if (ui.mode === 'slate') {
      vctx.fillStyle = '#000';
      vctx.fillRect(0, 0, W, H);
      centerText(ui.slate || 'Done', H / 2, 36, '#e8eaed', 'bold');
      return;
    }

    if (ui.mode === 'replay' && replay) {
      const n = replay.imgs.length;
      if (!replay.paused) {
        replay.pos += (dt / 1000) * FPS * replay.speed;
        if (replay.pos >= n) replay.pos -= n;
        $('scrub').value = String(Math.floor(replay.pos));
      }
      const i = Math.max(0, Math.min(n - 1, Math.floor(replay.pos)));
      vctx.drawImage(replay.imgs[i], 0, 0, W, H);
      if ($('cSkel').checked && replay.overlay && replay.overlay[i]) {
        drawSkeleton(vctx, replay.overlay[i]);
      }
      vctx.fillStyle = 'rgba(0,0,0,0.55)';
      vctx.fillRect(W - 150, 8, 142, 26);
      vctx.fillStyle = '#e8eaed';
      vctx.font = '15px system-ui,Segoe UI,Roboto,sans-serif';
      vctx.textAlign = 'right';
      vctx.textBaseline = 'middle';
      vctx.fillText(`replay ${i + 1}/${n}  ${replay.speed}x`, W - 14, 21);
      return;
    }

    // live preview (also used during flash / countdown / recording)
    drawCap();
    if (CFG.mirrorPreview) {
      vctx.save();
      vctx.translate(W, 0);
      vctx.scale(-1, 1);
      vctx.drawImage(cap, 0, 0, W, H);
      vctx.restore();
    } else {
      vctx.drawImage(cap, 0, 0, W, H);
    }

    // persistent word label (top-left)
    vctx.fillStyle = 'rgba(0,0,0,0.55)';
    vctx.fillRect(8, 8, 220, 30);
    vctx.fillStyle = '#e8eaed';
    vctx.font = 'bold 18px system-ui,Segoe UI,Roboto,sans-serif';
    vctx.textAlign = 'left';
    vctx.textBaseline = 'middle';
    vctx.fillText(CFG.wordDisplay, 16, 24, 204);

    if (ui.banner) {
      vctx.fillStyle = 'rgba(0,0,0,0.6)';
      vctx.fillRect(0, 0, W, H);
      centerText(ui.banner.text, H * 0.42, 96, '#ffffff', 'bold');
      if (ui.banner.sub) centerText(ui.banner.sub, H * 0.72, 26, '#bdc1c6', 'normal');
    }
    if (ui.count !== null) {
      vctx.fillStyle = 'rgba(0,0,0,0.35)';
      vctx.fillRect(0, 0, W, H);
      centerText(String(ui.count), H * 0.5, 180, '#ffffff', 'bold');
    }
    if (ui.rec) {
      // red dot + frame counter in the top-right corner, progress bar at the bottom
      vctx.fillStyle = 'rgba(0,0,0,0.6)';
      vctx.fillRect(W - 190, 8, 182, 34);
      vctx.fillStyle = '#ea4335';
      vctx.beginPath();
      vctx.arc(W - 172, 25, 8, 0, 6.2832);
      vctx.fill();
      vctx.fillStyle = '#ffffff';
      vctx.font = 'bold 20px system-ui,Segoe UI,Roboto,sans-serif';
      vctx.textAlign = 'right';
      vctx.textBaseline = 'middle';
      vctx.fillText(`REC ${ui.rec.n}/${N}`, W - 16, 26);
      vctx.fillStyle = '#ea4335';
      vctx.fillRect(0, H - 6, (W * ui.rec.n) / N, 6);
    }
  }

  // ----------------------------------------------------- phases
  let note = ''; // one-shot confirmation shown on the next take's banner ("Saved ..." / "Discarded ...")

  async function flashAndCountdown(takeNo, saved) {
    ui.banner = { text: CFG.wordDisplay, sub: `${note}take ${takeNo} of ${CFG.target}  -  get ready` };
    say(note + 'Get ready to sign: ' + CFG.wordDisplay, note.startsWith('Saved') ? '#81c995' : undefined);
    note = '';
    await sleep(CFG.flashMs);
    ui.banner = null;
    for (let c = CFG.countdown; c >= 1; c--) {
      if (S.stop) break;
      ui.count = c;
      await sleep(CFG.stepMs);
    }
    ui.count = null;
  }

  async function recordClip() {
    const frames = new Array(N);
    const stamps = [];
    const pres = [];
    ui.rec = { n: 0 };
    say('Recording - sign now!');
    const t0 = performance.now();
    for (let i = 0; i < N; i++) {
      if (S.stop) {
        ui.rec = null;
        return null;
      }
      const wait = t0 + (i * 1000) / FPS - performance.now();
      if (wait > 0) await sleep(wait);
      drawCap();
      frames[i] = cap.toDataURL('image/jpeg', CFG.jpegQuality);
      stamps.push(performance.now() - t0);
      pres.push(presented);
      ui.rec.n = i + 1;
    }
    await sleep(150); // let the final N/N counter be seen before the overlay clears
    ui.rec = null;
    let dups = -1;
    if (presentedSupported) {
      dups = 0;
      for (let i = 1; i < pres.length; i++) if (pres[i] === pres[i - 1]) dups++;
    }
    return {
      frames,
      meta: {
        ts: stamps.map((v) => Math.round(v * 10) / 10),
        dup_frames: dups,
        camera_fps: camFps,
        camera_w: video.videoWidth,
        camera_h: video.videoHeight,
      },
    };
  }

  async function sendAndAnalyze(takeId, frames, meta) {
    await call('asl.begin', [takeId]);
    const CH = 6;
    for (let s = 0; s < frames.length; s += CH) {
      const chunk = frames.slice(s, s + CH).map((u) => u.slice(u.indexOf(',') + 1));
      say(`Uploading frames ${Math.min(s + CH, frames.length)}/${frames.length} ...`);
      await call('asl.upload', [takeId, s, ...chunk]);
    }
    say('Analyzing hands / pose / face with MediaPipe (a few seconds) ...');
    return await call('asl.process', [takeId, JSON.stringify(meta)]);
  }

  const pct = (v) => Math.round(v * 100) + '%';

  function describeQA(qa, meta) {
    const lines = [];
    lines.push(
      `<b>Detection</b> - left hand ${pct(qa.lh_cov)} | right hand ${pct(qa.rh_cov)} | ` +
        `either hand ${pct(qa.any_hand_cov)} | pose ${pct(qa.pose_cov)} | face ${pct(qa.face_cov)}`
    );
    const bits = [`${qa.measured_fps.toFixed(1)} fps measured`];
    if (meta.dup_frames >= 0) bits.push(`${meta.dup_frames} repeated frames`);
    if (meta.camera_fps) bits.push(`camera reports ${Math.round(meta.camera_fps)} fps`);
    bits.push(`max gap ${Math.round(qa.max_gap_ms)} ms`);
    lines.push(`<b>Capture</b> - ${bits.join(' | ')}`);
    const warn = [];
    if (qa.low_hands) {
      warn.push(
        `Hands were visible in only ${pct(qa.any_hand_cov)} of frames (below ${pct(CFG.minHandCoverage)}) - probably retry.`
      );
    }
    if (qa.stuttered) {
      warn.push('Capture stuttered (low fps or many repeated frames) - check lighting / close other camera apps.');
    }
    if (warn.length) lines.push(`<span style="color:#fdd663;">&#9888; ${warn.join('<br>&#9888; ')}</span>`);
    return lines.join('<br>');
  }

  function review(frames, res, meta) {
    return new Promise(async (resolve, reject) => {
      let imgs;
      try {
        imgs = await Promise.all(
          frames.map((u) => {
            const im = new Image();
            im.src = u;
            return im.decode().then(() => im);
          })
        );
      } catch (e) {
        reject(e);
        return;
      }
      replay = { imgs, overlay: res.overlay, pos: 0, speed: 1, paused: false };
      $('qa').innerHTML = describeQA(res.qa, meta);
      $('scrub').value = '0';
      $('bPlay').textContent = 'Pause';
      root.querySelectorAll('.spd').forEach((b) => b.classList.toggle('on', b.dataset.s === '1'));
      $('live').style.display = 'none';
      $('review').style.display = 'block';
      ui.mode = 'replay';
      say('Replaying your take. Was it good?');

      const done = (decision) => {
        ui.mode = 'live';
        replay = null;
        $('review').style.display = 'none';
        $('live').style.display = 'flex';
        resolve(decision);
      };
      $('bGood').onclick = () => done('good');
      $('bBad').onclick = () => done('bad');
      $('bStop').onclick = () => {
        S.stop = true;
        done('stop');
      };
      $('bPlay').onclick = () => {
        replay.paused = !replay.paused;
        $('bPlay').textContent = replay.paused ? 'Play' : 'Pause';
      };
      $('bRestart').onclick = () => {
        replay.pos = 0;
        replay.paused = false;
        $('bPlay').textContent = 'Pause';
      };
      root.querySelectorAll('.spd').forEach((b) => {
        b.onclick = () => {
          replay.speed = parseFloat(b.dataset.s);
          root.querySelectorAll('.spd').forEach((x) => x.classList.toggle('on', x === b));
        };
      });
      $('scrub').oninput = () => {
        replay.paused = true;
        $('bPlay').textContent = 'Play';
        replay.pos = parseFloat($('scrub').value);
      };
    });
  }

  function waitForStartButton() {
    return new Promise((resolve) => {
      $('live').innerHTML = '';
      const go = document.createElement('button');
      go.className = 'aslb good';
      go.textContent = 'Start next take';
      go.onclick = () => resolve('go');
      const stop = document.createElement('button');
      stop.className = 'aslb';
      stop.textContent = 'Stop session';
      stop.onclick = () => {
        S.stop = true;
        resolve('stop');
      };
      $('live').append(go, stop);
      say('Ready when you are.');
    });
  }

  function showLiveControls() {
    $('live').innerHTML = '';
    const stop = document.createElement('button');
    stop.className = 'aslb';
    stop.textContent = 'Stop session';
    stop.onclick = () => {
      S.stop = true;
      stop.disabled = true;
      say('Stopping after the current step ...');
    };
    $('live').append(stop);
  }

  // ------------------------------------------------------- main
  try {
    $('hdrWord').textContent = CFG.wordDisplay;
    say('Starting camera - click Allow when the browser asks ...');
    const init = await call('asl.init', []);
    conn = { pose: init.pose_conn, hand: init.hand_conn, faceStride: init.face_stride };
    let saved = init.saved;
    $('hdrCount').textContent = `${saved} / ${CFG.target} saved`;
    if (saved >= CFG.target) {
      say(
        `Already have ${saved} takes for "${CFG.word}" (${CFG.person}). ` +
          'Raise TAKES in the config cell if you want more.',
        '#fdd663'
      );
      ui.mode = 'slate';
      ui.slate = 'Nothing to do - target already reached';
      requestAnimationFrame(render);
      return;
    }

    await startCamera();
    requestAnimationFrame(render);

    let takeId = 0;
    let first = true;
    while (!S.stop && saved < CFG.target) {
      takeId++;
      $('hdrCount').textContent = `take ${saved + 1} / ${CFG.target}  (${saved} saved)`;
      if (!first && !CFG.autoContinue) {
        await waitForStartButton();
        if (S.stop) break;
      }
      first = false;
      showLiveControls();
      await flashAndCountdown(saved + 1, saved);
      if (S.stop) break;
      const rec = await recordClip();
      if (!rec) break;
      $('live').innerHTML = '';
      const res = await sendAndAnalyze(takeId, rec.frames, rec.meta);
      const decision = await review(rec.frames, res, rec.meta);
      if (decision === 'good') {
        say('Saving ...');
        const r = await call('asl.save', [takeId]);
        saved = r.saved;
        note = `Saved ${r.file} (${saved}/${CFG.target}).  `;
        say(note, '#81c995');
      } else {
        await call('asl.discard', [takeId]);
        if (decision === 'bad') {
          note = 'Discarded - redoing.  ';
          say(note, '#fdd663');
        }
      }
    }

    stopCamera();
    const fin = await call('asl.finish', []);
    $('live').innerHTML = '';
    ui.mode = 'slate';
    ui.slate = `Session finished - ${fin.saved} takes saved`;
    $('hdrCount').textContent = `${fin.saved} / ${CFG.target} saved`;
    say(`Done. ${fin.saved} takes saved in ${fin.out_dir}. Run the next cell for a summary / zip download.`, '#81c995');
  } catch (err) {
    stopCamera();
    fail(err);
  }
})();
