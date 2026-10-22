(function (root) {
  "use strict";

  const LIMITS = Object.freeze({
    maxAhead: 2,
    keepBehind: 3,
    catchUpAfter: 1,
    liveMargin: 0.15,
    initialBuffer: 0.1,
    appendBytes: 256 * 1024,
    pollMs: 100,
    operationTimeoutMs: 15000
  });

  function rangesOf(buffered) {
    const ranges = [];
    for (let i = 0; i < buffered.length; i += 1) {
      const start = buffered.start(i);
      const end = buffered.end(i);
      if (Number.isFinite(start) && Number.isFinite(end) && end > start) ranges.push([start, end]);
    }
    return ranges;
  }

  function liveEdgeTarget(ranges, currentTime, initial) {
    if (!ranges.length || !Number.isFinite(currentTime)) return null;
    const [start, end] = ranges[ranges.length - 1];
    if (initial && end - start + 1e-6 < LIMITS.initialBuffer) return null;
    if (!initial && end - currentTime <= LIMITS.catchUpAfter) return null;
    // Stay inside the latest buffered range, including after a timestamp gap.
    return Math.min(end - Math.min(0.001, (end - start) / 2), Math.max(start, end - LIMITS.liveMargin));
  }

  function bufferedAhead(ranges, currentTime) {
    if (!ranges.length) return 0;
    return Math.max(0, ranges[ranges.length - 1][1] - Math.max(currentTime, ranges[0][0]));
  }

  function trimEnd(ranges, currentTime) {
    const end = currentTime - LIMITS.keepBehind;
    return ranges.length && end - ranges[0][0] >= 0.5 ? end : null;
  }

  function abortError() {
    const error = new Error("Playback stopped.");
    error.name = "AbortError";
    return error;
  }

  function checkAbort(signal) {
    if (signal.aborted) throw abortError();
  }

  // Install listeners before invoking an operation, and remove them on every exit.
  function waitForEvent(target, success, signal, failures, action) {
    return new Promise(function (resolve, reject) {
      let timer;
      const finish = function (error) {
        clearTimeout(timer);
        target.removeEventListener(success, done);
        failures.forEach(function (name) { target.removeEventListener(name, failed); });
        signal.removeEventListener("abort", cancelled);
        if (error) reject(error); else resolve();
      };
      const done = function () { finish(); };
      const failed = function (event) { finish(new Error("Media operation failed: " + event.type + ".")); };
      const cancelled = function () { finish(abortError()); };
      if (signal.aborted) { reject(abortError()); return; }
      target.addEventListener(success, done);
      failures.forEach(function (name) { target.addEventListener(name, failed); });
      signal.addEventListener("abort", cancelled, { once: true });
      timer = setTimeout(function () { finish(new Error("Timed out waiting for " + success + ".")); }, LIMITS.operationTimeoutMs);
      try { if (action) action(); } catch (error) { finish(error); }
    });
  }

  async function sourceBufferOperation(sourceBuffer, action, signal) {
    while (sourceBuffer.updating) {
      await waitForEvent(sourceBuffer, "updateend", signal, ["error", "abort"]);
    }
    checkAbort(signal);
    await waitForEvent(sourceBuffer, "updateend", signal, ["error", "abort"], action);
    checkAbort(signal);
  }

  function delay(ms, signal) {
    return new Promise(function (resolve, reject) {
      if (signal.aborted) { reject(abortError()); return; }
      const cancelled = function () {
        clearTimeout(timer);
        signal.removeEventListener("abort", cancelled);
        reject(abortError());
      };
      const timer = setTimeout(function () {
        signal.removeEventListener("abort", cancelled);
        resolve();
      }, ms);
      signal.addEventListener("abort", cancelled, { once: true });
    });
  }

  async function waitForBufferRoom(sourceBuffer, video, signal) {
    checkAbort(signal);
    while (bufferedAhead(rangesOf(sourceBuffer.buffered), video.currentTime) >= LIMITS.maxAhead) {
      await delay(LIMITS.pollMs, signal);
    }
    checkAbort(signal);
  }

  // One read result at a time, no application queue; also gate each slice of a
  // coalesced network chunk so a large read cannot bypass playback backpressure.
  async function pumpStream(reader, sourceBuffer, video, signal, afterAppend) {
    for (;;) {
      await waitForBufferRoom(sourceBuffer, video, signal);
      const result = await reader.read();
      checkAbort(signal);
      if (result.done) throw new Error("The camera stream ended. Use Start to reconnect.");
      const bytes = result.value;
      if (!(bytes instanceof Uint8Array)) throw new Error("The stream returned invalid byte data.");
      for (let offset = 0; offset < bytes.byteLength; offset += LIMITS.appendBytes) {
        await waitForBufferRoom(sourceBuffer, video, signal);
        const slice = bytes.subarray(offset, Math.min(bytes.byteLength, offset + LIMITS.appendBytes));
        await sourceBufferOperation(sourceBuffer, function () { sourceBuffer.appendBuffer(slice); }, signal);
        await afterAppend();
        checkAbort(signal);
      }
    }
  }

  function readQuality(video) {
    if (typeof video.getVideoPlaybackQuality !== "function") return null;
    try {
      const quality = video.getVideoPlaybackQuality();
      if (!Number.isFinite(quality.totalVideoFrames) || !Number.isFinite(quality.droppedVideoFrames)) return null;
      return {
        total: quality.totalVideoFrames,
        dropped: quality.droppedVideoFrames,
        // totalVideoFrames includes frames dropped before decoding. Report an
        // actual decoded count separately only when the browser exposes one.
        decoded: Number.isFinite(video.webkitDecodedFrameCount) ? video.webkitDecodedFrameCount : null
      };
    } catch (_) { return null; }
  }

  class PlaybackStats {
    constructor() { this.reset(); }
    reset() { this.baseline = null; }
    sample(now, quality, playing, hidden) {
      if (hidden || !playing || !quality) {
        this.reset();
        return { state: hidden ? "hidden" : !playing ? "inactive" : "unsupported" };
      }
      if (!this.baseline || quality.total < this.baseline.quality.total || quality.dropped < this.baseline.quality.dropped) {
        this.baseline = { time: now, quality: quality };
      }
      const seconds = (now - this.baseline.time) / 1000;
      if (seconds < 0.25) return { state: "warming" };
      const total = Math.max(0, quality.total - this.baseline.quality.total);
      const dropped = Math.max(0, quality.dropped - this.baseline.quality.dropped);
      const baseDecoded = this.baseline.quality.decoded;
      return {
        state: "playing", seconds: seconds,
        total: total / seconds,
        rendered: Math.max(0, total - dropped) / seconds,
        dropped: dropped / seconds,
        decoded: quality.decoded !== null && baseDecoded !== null && quality.decoded >= baseDecoded
          ? (quality.decoded - baseDecoded) / seconds : null
      };
    }
  }

  class H264Player {
    constructor(elements, environment) {
      this.elements = elements;
      this.env = environment;
      this.session = null;
      this.updateButtons();
    }

    updateButtons() {
      this.elements.start.disabled = !!this.session;
      this.elements.stop.disabled = !this.session;
    }

    status(message, error) {
      this.elements.status.textContent = message;
      this.elements.status.dataset.state = error ? "error" : "normal";
    }

    listen(session, target, name, handler) {
      target.addEventListener(name, handler);
      session.listeners.push(function () { target.removeEventListener(name, handler); });
    }

    start() {
      if (this.session) return this.session.done;
      if (!this.env.MediaSource || typeof this.env.MediaSource.isTypeSupported !== "function") {
        this.status("This browser does not support MediaSource. Try a browser with H.264 MP4 MediaSource support.", true);
        return Promise.resolve();
      }
      if (!this.env.fetch || !this.env.AbortController) {
        this.status("This browser does not support streaming fetch and cancellation.", true);
        return Promise.resolve();
      }
      const session = {
        controller: new this.env.AbortController(), listeners: [], reader: null,
        response: null, mediaSource: null, sourceBuffer: null, url: null, timer: null,
        started: false, playing: false, stats: new PlaybackStats()
      };
      this.session = session;
      this.updateButtons();
      this.status("Connecting to /stream.mp4…");
      this.elements.metadata.textContent = "Waiting for stream information.";
      this.elements.stats.textContent = "Browser FPS: waiting for playback.";
      session.done = this.run(session).catch((error) => {
        if (this.session !== session || session.controller.signal.aborted) return;
        const explanation = error.name === "QuotaExceededError"
          ? "The browser's video buffer is full. Use Start to reconnect."
          : error.message || String(error);
        this.stop("Error: " + explanation, true);
      });
      return session.done;
    }

    async run(session) {
      const signal = session.controller.signal;
      const video = this.elements.video;
      const response = await this.env.fetch("/stream.mp4", { signal: signal, cache: "no-store", mode: "same-origin", redirect: "error" });
      // A late fetch resolution belongs to the old session, never the new video.
      if (signal.aborted) {
        if (response.body) await response.body.cancel().catch(function () {});
        throw abortError();
      }
      session.response = response;
      if (!response.ok) throw new Error("Camera HTTP " + response.status + ". Use Start to retry.");
      if (!response.body || typeof response.body.getReader !== "function") throw new Error("The browser did not provide a readable video stream.");
      session.reader = response.body.getReader();
      const mime = (response.headers.get("Content-Type") || "").trim();
      if (!/^video\/mp4\s*;/i.test(mime) || !/;\s*codecs\s*=\s*"?avc1\.[0-9a-f]{6}"?(?:\s*;|\s*$)/i.test(mime)) {
        throw new Error("The camera must send Content-Type: video/mp4; codecs=\"avc1.xxxxxx\" using its actual SPS. Received: " + (mime || "(missing)") + ".");
      }
      if (!this.env.MediaSource.isTypeSupported(mime)) throw new Error("This browser cannot play the camera codec: " + mime + ".");
      const info = [mime];
      const width = response.headers.get("X-Video-Width");
      const height = response.headers.get("X-Video-Height");
      const fps = response.headers.get("X-Video-FPS");
      if (width && height) info.push(width + " × " + height);
      if (fps) info.push("camera target " + fps + " fps");
      this.elements.metadata.textContent = info.join(" · ");

      const mediaSource = session.mediaSource = new this.env.MediaSource();
      session.url = this.env.URL.createObjectURL(mediaSource);
      const fail = (message) => { if (this.session === session) this.stop("Error: " + message + " Use Start to reconnect.", true); };
      this.listen(session, video, "error", function () {
        const error = video.error;
        fail("Video decoding failed" + (error ? " (code " + error.code + ": " + (error.message || "media error") + ")." : "."));
      });
      this.listen(session, mediaSource, "sourceclose", function () { fail("The MediaSource closed unexpectedly."); });
      this.listen(session, mediaSource, "sourceended", function () { fail("The MediaSource ended unexpectedly."); });
      const refresh = (playing) => {
        if (this.session !== session) return;
        session.playing = playing;
        if (!playing) session.stats.reset();
        this.status(playing ? "Playing live." : video.paused && session.started ? "Paused. Forward buffering is limited." : "Buffering live video…");
        this.updateStats(session);
      };
      this.listen(session, video, "playing", function () { refresh(true); });
      ["pause", "waiting", "stalled", "seeking"].forEach((name) => this.listen(session, video, name, function () { refresh(false); }));
      this.listen(session, video, "seeked", function () { refresh(!video.paused && video.readyState >= 3); });
      this.listen(session, this.env.document, "visibilitychange", () => {
        session.stats.reset();
        this.updateStats(session);
      });
      video.muted = true;
      video.autoplay = false;
      video.pause();
      await waitForEvent(mediaSource, "sourceopen", signal, ["sourceclose"], function () {
        video.src = session.url;
        video.load();
      });
      checkAbort(signal);
      mediaSource.duration = Infinity;
      const sourceBuffer = session.sourceBuffer = mediaSource.addSourceBuffer(mime);
      this.listen(session, sourceBuffer, "error", function () { fail("The camera sent invalid or unsupported fragmented MP4 data."); });
      session.timer = this.env.setInterval(() => this.updateStats(session), 1000);
      this.status("Buffering live video…");
      await pumpStream(session.reader, sourceBuffer, video, signal, () => this.afterAppend(session));
    }

    async afterAppend(session) {
      const signal = session.controller.signal;
      checkAbort(signal);
      const video = this.elements.video;
      const sourceBuffer = session.sourceBuffer;
      const ranges = rangesOf(sourceBuffer.buffered);
      const initial = !session.started;
      const target = liveEdgeTarget(ranges, video.currentTime, initial);
      if (target !== null && (initial || (!video.paused && !video.seeking))) video.currentTime = target;
      if (initial && target !== null) {
        session.started = true;
        video.autoplay = true;
        // A stalled play() promise must not hold up the append loop. A rejection
        // still terminates the session and is visible to the user.
        Promise.resolve(video.play()).catch((error) => {
          if (this.session === session && !signal.aborted) this.stop("Error: Playback could not start: " + error.message + ". Use Start to retry.", true);
        });
      }
      const end = trimEnd(ranges, video.currentTime);
      if (end !== null) {
        await sourceBufferOperation(sourceBuffer, function () { sourceBuffer.remove(0, end); }, signal);
      }
    }

    updateStats(session) {
      if (this.session !== session) return;
      const video = this.elements.video;
      const result = session.stats.sample(this.env.performance.now(), readQuality(video),
        session.started && session.playing && !video.paused && !video.seeking, this.env.document.hidden);
      let text;
      if (result.state === "hidden") text = "Browser FPS: tab hidden; background playback is excluded.";
      else if (result.state === "inactive") text = "Browser FPS: waiting for active playback.";
      else if (result.state === "unsupported") text = "Browser FPS: getVideoPlaybackQuality is unavailable in this browser.";
      else if (result.state === "warming") text = "Browser FPS: measuring visible playback…";
      else {
        text = "Browser FPS: total " + result.total.toFixed(1) + " · rendered " + result.rendered.toFixed(1)
          + " · dropped " + result.dropped.toFixed(1);
        if (result.decoded !== null) text += " · decoded " + result.decoded.toFixed(1);
        text += " (" + result.seconds.toFixed(1) + " s visible playback window)";
      }
      this.elements.stats.textContent = text;
    }

    stop(message, error) {
      const session = this.session;
      this.session = null;
      if (session) {
        session.controller.abort();
        session.listeners.forEach(function (remove) { remove(); });
        this.env.clearInterval(session.timer);
        if (session.reader) {
          // cancel() also resolves a pending read; release only after it settles.
          Promise.resolve(session.reader.cancel()).catch(function () {}).finally(function () {
            try { session.reader.releaseLock(); } catch (_) {}
          });
        } else if (session.response && session.response.body) {
          Promise.resolve(session.response.body.cancel()).catch(function () {});
        }
        if (session.sourceBuffer && session.mediaSource.readyState === "open" && session.sourceBuffer.updating) {
          try { session.sourceBuffer.abort(); } catch (_) {}
        }
      }
      const video = this.elements.video;
      video.pause();
      video.autoplay = false;
      video.removeAttribute("src");
      video.load();
      if (session && session.url) this.env.URL.revokeObjectURL(session.url);
      this.elements.stats.textContent = "Browser FPS: stopped.";
      this.status(message || "Stopped. Use Start to connect.", error);
      this.updateButtons();
    }
  }

  function mount(document, environment) {
    const elements = {};
    ["video", "start", "stop", "status", "metadata", "stats"].forEach(function (id) { elements[id] = document.getElementById(id); });
    if (Object.values(elements).some(function (element) { return !element; })) return null;
    const player = new H264Player(elements, environment);
    const start = function () { player.start(); };
    const stop = function () { player.stop(); };
    const hide = function () { player.stop("Stopped because the page was closed or hidden. Use Start to reconnect."); };
    elements.start.addEventListener("click", start);
    elements.stop.addEventListener("click", stop);
    environment.addEventListener("pagehide", hide);
    environment.addEventListener("beforeunload", hide);
    player.destroy = function () {
      player.stop();
      elements.start.removeEventListener("click", start);
      elements.stop.removeEventListener("click", stop);
      environment.removeEventListener("pagehide", hide);
      environment.removeEventListener("beforeunload", hide);
    };
    player.start();
    return player;
  }

  const helpers = { LIMITS, rangesOf, liveEdgeTarget, bufferedAhead, trimEnd, sourceBufferOperation,
    waitForBufferRoom, pumpStream, readQuality, PlaybackStats, H264Player, mount };
  if (typeof module === "object" && module.exports) module.exports = helpers;
  if (root.document) {
    if (root.document.readyState === "loading") {
      root.document.addEventListener("DOMContentLoaded", function () { mount(root.document, root); }, { once: true });
    } else mount(root.document, root);
  }
})(typeof window !== "undefined" ? window : globalThis);
