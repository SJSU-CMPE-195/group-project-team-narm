import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { setImmediate as nextTurn, setTimeout as sleep } from 'node:timers/promises';

const require = createRequire(import.meta.url);
const {
  LIMITS, rangesOf, liveEdgeTarget, bufferedAhead, trimEnd, sourceBufferOperation,
  waitForBufferRoom, pumpStream, PlaybackStats, H264Player, mount
} = require('../main/h264_player.js');

function timeRanges(ranges) {
  return { length: ranges.length, start: (i) => ranges[i][0], end: (i) => ranges[i][1] };
}

class TrackedTarget extends EventTarget {
  listeners = new Map();
  addEventListener(name, handler, options) {
    if (!this.listeners.has(name)) this.listeners.set(name, new Set());
    this.listeners.get(name).add(handler);
    super.addEventListener(name, handler, options);
  }
  removeEventListener(name, handler, options) {
    this.listeners.get(name)?.delete(handler);
    super.removeEventListener(name, handler, options);
  }
  get listenerCount() { return [...this.listeners.values()].reduce((count, handlers) => count + handlers.size, 0); }
  emit(name) { this.dispatchEvent(new Event(name)); }
}

class MockSourceBuffer extends TrackedTarget {
  constructor({ autoComplete = false, secondsPerAppend = 0.2 } = {}) {
    super();
    this.autoComplete = autoComplete;
    this.secondsPerAppend = secondsPerAppend;
    this.ranges = [];
    this.updating = false;
    this.actions = [];
    this.pending = null;
    this.abortCalls = 0;
  }
  get buffered() { return timeRanges(this.ranges); }
  begin(action) {
    assert.equal(this.updating, false, 'append/remove must never overlap');
    this.actions.push(action);
    this.pending = action;
    this.updating = true;
    if (this.autoComplete) queueMicrotask(() => this.finish());
  }
  appendBuffer(bytes) { this.begin({ type: 'append', bytes: new Uint8Array(bytes) }); }
  remove(start, end) { this.begin({ type: 'remove', start, end }); }
  finish() {
    if (!this.updating) return;
    const action = this.pending;
    if (action?.type === 'append' && this.secondsPerAppend !== null) {
      const end = this.ranges.length ? this.ranges.at(-1)[1] : 0;
      this.ranges = [[this.ranges[0]?.[0] ?? 0, end + this.secondsPerAppend]];
    } else if (action?.type === 'remove') {
      this.ranges = this.ranges.flatMap(([start, end]) => {
        if (end <= action.start || start >= action.end) return [[start, end]];
        const pieces = [];
        if (start < action.start) pieces.push([start, action.start]);
        if (end > action.end) pieces.push([action.end, end]);
        return pieces;
      });
    }
    this.pending = null;
    this.updating = false;
    this.emit('updateend');
  }
  abort() {
    this.abortCalls += 1;
    this.pending = null;
    this.updating = false;
    this.emit('abort');
    this.emit('updateend');
  }
}

class MockReader {
  constructor(chunks = []) {
    this.chunks = [...chunks];
    this.readCalls = 0;
    this.cancelCalls = 0;
    this.releaseCalls = 0;
    this.pending = null;
    this.cancelled = false;
  }
  read() {
    assert.equal(this.pending, null, 'only one read may be outstanding');
    this.readCalls += 1;
    if (this.cancelled) return Promise.resolve({ done: true });
    if (this.chunks.length) return Promise.resolve(this.chunks.shift());
    return new Promise((resolve, reject) => { this.pending = { resolve, reject }; });
  }
  cancel() {
    this.cancelCalls += 1;
    this.cancelled = true;
    this.pending?.resolve({ done: true });
    this.pending = null;
    return Promise.resolve();
  }
  releaseLock() { this.releaseCalls += 1; }
  fail(error) {
    assert.ok(this.pending);
    this.pending.reject(error);
    this.pending = null;
  }
}

async function until(predicate, message = 'condition was not reached') {
  for (let i = 0; i < 100; i += 1) {
    if (predicate()) return;
    await nextTurn();
  }
  assert.fail(message);
}

function harness(options = {}) {
  const env = new TrackedTarget();
  env.document = new TrackedTarget();
  env.document.hidden = false;
  env.AbortController = AbortController;
  env.clock = 1000;
  env.performance = { now: () => env.clock };
  env.intervals = new Map();
  env.setInterval = (handler) => { const id = Symbol(); env.intervals.set(id, handler); return id; };
  env.clearInterval = (id) => env.intervals.delete(id);
  env.mediaSources = [];
  env.supportChecks = [];
  env.buffers = [];
  env.urls = new Map();
  env.revoked = [];
  env.URL = {
    createObjectURL: (source) => { const url = 'blob:mock-' + env.urls.size; env.urls.set(url, source); return url; },
    revokeObjectURL: (url) => env.revoked.push(url)
  };
  env.MediaSource = class extends TrackedTarget {
    static isTypeSupported(mime) { env.supportChecks.push(mime); return options.supported !== false; }
    constructor() { super(); this.readyState = 'closed'; env.mediaSources.push(this); }
    addSourceBuffer(mime) {
      this.mime = mime;
      this.sourceBuffer = new MockSourceBuffer({ autoComplete: options.autoComplete !== false, secondsPerAppend: options.secondsPerAppend ?? 0.2 });
      env.buffers.push(this.sourceBuffer);
      return this.sourceBuffer;
    }
  };
  class MockVideo extends TrackedTarget {
    constructor() {
      super();
      this.currentTime = 0;
      this.paused = true;
      this.seeking = false;
      this.readyState = 0;
      this.playCalls = 0;
      this.loadCalls = 0;
      this.quality = { totalVideoFrames: 0, droppedVideoFrames: 0 };
      this.src = '';
    }
    load() {
      this.loadCalls += 1;
      const source = env.urls.get(this.src);
      if (source && source.readyState === 'closed' && options.open !== false) {
        queueMicrotask(() => {
          if (env.urls.get(this.src) !== source) return;
          source.readyState = 'open';
          source.emit('sourceopen');
        });
      } else if (!source) {
        this.currentTime = 0;
        this.readyState = 0;
        this.quality = { totalVideoFrames: 0, droppedVideoFrames: 0 };
      }
    }
    pause() { const wasPaused = this.paused; this.paused = true; if (!wasPaused) this.emit('pause'); }
    play() {
      this.playCalls += 1;
      if (options.playError) return Promise.reject(options.playError);
      this.paused = false;
      this.readyState = 4;
      this.emit('playing');
      return options.pendingPlay ? new Promise(() => {}) : Promise.resolve();
    }
    removeAttribute(name) { if (name === 'src') this.src = ''; }
    getVideoPlaybackQuality() { return this.quality; }
  }
  const elements = { video: new MockVideo() };
  for (const name of ['start', 'stop', 'status', 'metadata', 'stats']) {
    elements[name] = new TrackedTarget();
    elements[name].dataset = {};
    elements[name].textContent = '';
  }
  env.document.getElementById = (id) => elements[id];
  env.reader = new MockReader(options.chunks ?? [{ value: new Uint8Array([1, 2, 3]), done: false }]);
  env.bodyCancelCalls = 0;
  const response = {
    ok: options.httpStatus === undefined || options.httpStatus === 200,
    status: options.httpStatus ?? 200,
    headers: new Headers(options.headers ?? { 'Content-Type': 'video/mp4; codecs="avc1.42E01E"', 'X-Video-Width': '640', 'X-Video-Height': '480', 'X-Video-FPS': '60' }),
    body: {
      getReader: () => env.reader,
      cancel: () => { env.bodyCancelCalls += 1; return env.reader.cancel(); }
    }
  };
  env.fetchCalls = [];
  env.fetch = (url, init) => {
    env.fetchCalls.push({ url, init });
    return options.fetch ? options.fetch(response) : Promise.resolve(response);
  };
  const player = new H264Player(elements, env);
  return { env, elements, player, response };
}

test('classic local script and HTML need no modules, CDNs, or secure-context decoder', () => {
  const script = readFileSync(new URL('../main/h264_player.js', import.meta.url), 'utf8');
  assert.doesNotThrow(() => runInNewContext(script, {}), 'document-free loading must be safe');
  const html = readFileSync(new URL('../main/h264_player.html', import.meta.url), 'utf8');
  assert.match(html, /<script src="\/player\.js" defer><\/script>/);
  assert.match(html, /<video[^>]*\bmuted\b[^>]*\bautoplay\b[^>]*\bplaysinline\b[^>]*\bcontrols\b/);
  assert.equal((html.match(/<video\b/g) ?? []).length, 1);
  assert.doesNotMatch(html, /(?:src|href)="https?:\/\//);
  assert.doesNotMatch(script, /\bVideoDecoder\b|\bWebCodecs\b/);
});

test('live edge starts after 0.1 s and seeks inside the latest range only when over 1 s behind', () => {
  assert.equal(liveEdgeTarget([], 0, true), null);
  assert.equal(liveEdgeTarget([[0, 0.09]], 0, true), null);
  assert.equal(liveEdgeTarget([[0, 0.1]], 0, true), 0);
  assert.equal(liveEdgeTarget([[0, 10]], 9, false), null);
  assert.equal(liveEdgeTarget([[0, 10]], 8.99, false), 9.85);
  assert.equal(liveEdgeTarget([[0, 1], [20, 20.1]], 1, false), 20);
  assert.equal(liveEdgeTarget([[4, 5]], 6, false), null);
  assert.equal(liveEdgeTarget([[0, 1]], NaN, true), null);
});

test('buffer helpers handle nonzero timestamps, gaps, and roughly 3 seconds of history', () => {
  assert.deepEqual(rangesOf(timeRanges([[2, 4], [5, 8]])), [[2, 4], [5, 8]]);
  assert.deepEqual(rangesOf(timeRanges([[0, Infinity], [2, 1], [3, 4]])), [[3, 4]]);
  assert.equal(bufferedAhead([[100, 100.5]], 0), 0.5);
  assert.equal(bufferedAhead([[2, 4], [5, 8]], 6), 2);
  assert.equal(bufferedAhead([], 0), 0);
  assert.equal(trimEnd([[0, 10]], 8), 5);
  assert.equal(trimEnd([[4.7, 10]], 8), null);
  assert.equal(trimEnd([[0, 2]], 2), null);
});

test('append waits for a busy buffer, and append/remove operations never overlap', async (t) => {
  const sb = new MockSourceBuffer();
  const controller = new AbortController();
  t.after(() => controller.abort());
  sb.updating = true;
  const append = sourceBufferOperation(sb, () => sb.appendBuffer(new Uint8Array([1])), controller.signal);
  assert.equal(sb.actions.length, 0);
  sb.finish();
  await until(() => sb.actions.length === 1);
  const remove = sourceBufferOperation(sb, () => sb.remove(0, 1), controller.signal);
  assert.equal(sb.actions.length, 1);
  sb.finish();
  await append;
  await until(() => sb.actions.length === 2);
  assert.equal(sb.actions[1].type, 'remove');
  sb.finish();
  await remove;
  assert.equal(sb.listenerCount, 0);
});

test('SourceBuffer cancellation and asynchronous/synchronous errors remove listeners', async () => {
  for (const outcome of ['cancel', 'error', 'throw']) {
    const sb = new MockSourceBuffer();
    const controller = new AbortController();
    const operation = sourceBufferOperation(sb, () => {
      if (outcome === 'throw') throw new Error('bad append');
      sb.appendBuffer(new Uint8Array([1]));
    }, controller.signal);
    const rejected = assert.rejects(operation, outcome === 'cancel' ? { name: 'AbortError' } : /bad append|Media operation failed: error/);
    if (outcome === 'cancel') controller.abort();
    if (outcome === 'error') sb.emit('error');
    await rejected;
    assert.equal(sb.listenerCount, 0);
  }
});

test('buffer pressure waits without reading and cancellation interrupts that wait', async () => {
  const sb = new MockSourceBuffer();
  sb.ranges = [[0, 4]];
  const controller = new AbortController();
  let ready = false;
  const room = waitForBufferRoom(sb, { currentTime: 0 }, controller.signal).then(() => { ready = true; });
  const rejected = assert.rejects(room, { name: 'AbortError' });
  await nextTurn();
  assert.equal(ready, false);
  controller.abort();
  await rejected;

  const moving = { currentTime: 0 };
  const second = new AbortController();
  const released = waitForBufferRoom(sb, moving, second.signal);
  moving.currentTime = 2.1;
  await released;
});

test('stream pump reads one chunk at a time and slices it without an unlimited queue', async (t) => {
  const sb = new MockSourceBuffer({ secondsPerAppend: null });
  const reader = new MockReader([{ value: new Uint8Array(LIMITS.appendBytes * 2 + 7), done: false }, { value: new Uint8Array([2]), done: false }]);
  const controller = new AbortController();
  t.after(() => { controller.abort(); reader.cancel(); });
  let maintenance = 0;
  const task = pumpStream(reader, sb, { currentTime: 0 }, controller.signal, async () => { maintenance += 1; });
  const rejected = assert.rejects(task, { name: 'AbortError' });
  for (let i = 0; i < 3; i += 1) {
    await until(() => sb.actions.length === i + 1);
    assert.equal(reader.readCalls, 1, 'do not read ahead while appending a chunk');
    assert.ok(sb.actions[i].bytes.byteLength <= LIMITS.appendBytes);
    sb.finish();
  }
  await until(() => reader.readCalls === 2);
  assert.equal(maintenance, 3);
  assert.deepEqual(sb.actions.slice(0, 3).map((action) => action.bytes.length), [LIMITS.appendBytes, LIMITS.appendBytes, 7]);
  controller.abort();
  await rejected;
});

test('coalesced chunks cannot bypass the forward buffer bound', async (t) => {
  const sb = new MockSourceBuffer({ autoComplete: true, secondsPerAppend: 2 });
  const reader = new MockReader([{ value: new Uint8Array(LIMITS.appendBytes * 3), done: false }]);
  const controller = new AbortController();
  t.after(() => { controller.abort(); reader.cancel(); });
  const task = pumpStream(reader, sb, { currentTime: 0 }, controller.signal, async () => {});
  const rejected = assert.rejects(task, { name: 'AbortError' });
  await until(() => sb.actions.length === 1 && !sb.updating);
  await sleep(120);
  assert.equal(sb.actions.length, 1);
  assert.equal(reader.readCalls, 1);
  controller.abort();
  await rejected;
});

test('FPS uses a stable actual playback window, excludes hidden/paused time, and subtracts drops', () => {
  const stats = new PlaybackStats();
  const quality = (total, dropped, decoded = total) => ({ total, dropped, decoded });
  assert.equal(stats.sample(10000, quality(900, 20), false, false).state, 'inactive');
  assert.equal(stats.sample(20000, quality(900, 20), true, false).state, 'warming');
  const sample = stats.sample(22000, quality(960, 24, 950), true, false);
  assert.equal(sample.total, 30);
  assert.equal(sample.rendered, 28);
  assert.equal(sample.dropped, 2);
  assert.equal(sample.decoded, 25);
  assert.equal(stats.sample(24000, quality(1020, 28), true, false).seconds, 4);
  assert.equal(stats.sample(25000, quality(1020, 28), true, true).state, 'hidden');
  assert.equal(stats.sample(80000, quality(1100, 100), true, false).state, 'warming');
  assert.equal(stats.sample(82000, quality(1120, 102), true, false).rendered, 9);
  assert.equal(stats.sample(83000, quality(1120, 102), false, false).state, 'inactive');
  assert.equal(stats.sample(90000, quality(1120, 102), true, false).state, 'warming');
  assert.equal(stats.sample(91000, quality(0, 0), true, false).state, 'warming', 'counter reset starts a fresh window');
  assert.equal(stats.sample(92000, null, true, false).state, 'unsupported');
});

test('session uses the exact response MIME, one fetch/video, and cancels/resets every resource', async (t) => {
  const { env, elements, player } = harness();
  t.after(() => player.stop());
  const task = player.start();
  assert.equal(player.start(), task, 'repeated Start cannot open a second fetch');
  await until(() => elements.video.playCalls === 1 && env.reader.pending);
  assert.deepEqual(env.supportChecks, ['video/mp4; codecs="avc1.42E01E"']);
  assert.equal(env.mediaSources[0].mime, env.supportChecks[0]);
  assert.equal(env.mediaSources[0].duration, Infinity);
  assert.equal(env.fetchCalls.length, 1);
  assert.equal(env.fetchCalls[0].url, '/stream.mp4');
  assert.equal(env.fetchCalls[0].init.cache, 'no-store');
  assert.match(elements.metadata.textContent, /640 × 480.*camera target 60 fps/);
  assert.equal(elements.start.disabled, true);
  elements.video.quality = { totalVideoFrames: 24, droppedVideoFrames: 4 };
  env.clock += 1000;
  [...env.intervals.values()][0]();
  assert.match(elements.stats.textContent, /total 24\.0 · rendered 20\.0 · dropped 4\.0/);
  env.document.hidden = true;
  env.document.emit('visibilitychange');
  assert.match(elements.stats.textContent, /tab hidden/);
  player.stop();
  await task;
  await nextTurn();
  assert.equal(env.fetchCalls[0].init.signal.aborted, true);
  assert.equal(env.reader.cancelCalls, 1);
  assert.equal(env.reader.releaseCalls, 1);
  assert.equal(env.revoked.length, 1);
  assert.equal(elements.video.src, '');
  assert.equal(elements.video.paused, true);
  assert.equal(elements.video.currentTime, 0);
  assert.equal(env.intervals.size, 0);
  for (const target of [elements.video, env.document, ...env.mediaSources, ...env.buffers]) assert.equal(target.listenerCount, 0);
  assert.equal(elements.start.disabled, false);
  assert.equal(elements.stop.disabled, true);
});

test('initial playback waits for 0.1 s and a pending play promise does not block appends', async (t) => {
  const { env, elements, player } = harness({
    autoComplete: false, secondsPerAppend: 0.04, pendingPlay: true,
    chunks: Array.from({ length: 4 }, () => ({ value: new Uint8Array([1]), done: false }))
  });
  t.after(() => player.stop());
  const task = player.start();
  for (let i = 0; i < 4; i += 1) {
    await until(() => env.buffers[0]?.actions.length === i + 1);
    assert.equal(elements.video.playCalls, i < 3 ? 0 : 1, '0.04 and 0.08 seconds are too short to start');
    env.buffers[0].finish();
  }
  await until(() => env.reader.readCalls === 5);
  assert.equal(elements.video.playCalls, 1);
  assert.equal(env.buffers[0].actions.length, 4);
  player.stop();
  await task;
});

test('post-append catchup and history removal are serialized', async (t) => {
  const { env, elements, player } = harness();
  t.after(() => player.stop());
  const task = player.start();
  await until(() => env.reader.pending);
  const session = player.session;
  const sb = env.buffers[0];
  sb.ranges = [[0, 10]];
  elements.video.currentTime = 8;
  await player.afterAppend(session);
  assert.equal(elements.video.currentTime, 9.85);
  const removal = sb.actions.at(-1);
  assert.equal(removal.type, 'remove');
  assert.ok(Math.abs(removal.end - 6.85) < 1e-6);
  assert.equal(sb.updating, false);
  elements.video.pause();
  sb.ranges = [[6.85, 15]];
  await player.afterAppend(session);
  assert.equal(elements.video.currentTime, 9.85, 'native pause must remain paused');
  player.stop();
  await task;
});

test('unsupported or missing codec and HTTP errors cancel the response and show readable errors', async () => {
  for (const options of [
    { supported: false },
    { headers: { 'Content-Type': 'video/mp4' } },
    { headers: {} },
    { httpStatus: 503 }
  ]) {
    const { env, elements, player } = harness(options);
    await player.start();
    assert.equal(player.session, null);
    assert.equal(elements.status.dataset.state, 'error');
    assert.match(elements.status.textContent, /cannot play|Content-Type|HTTP 503/);
    assert.equal(env.reader.cancelCalls, 1);
    assert.equal(env.mediaSources.length, 0);
    assert.equal(elements.start.disabled, false);
  }
});

test('EOF, network, decoder, SourceBuffer, and autoplay errors permit a manual reconnect', async (t) => {
  for (const kind of ['eof', 'network', 'video', 'sourcebuffer', 'sourceclose', 'play']) {
    const h = harness(kind === 'eof' ? { chunks: [{ done: true }] } : kind === 'play' ? { playError: new Error('autoplay blocked') } : {});
    const { env, elements, player } = h;
    t.after(() => player.stop());
    const task = player.start();
    if (kind !== 'eof' && kind !== 'play') {
      await until(() => env.reader.pending);
      if (kind === 'network') env.reader.fail(new Error('connection lost'));
      if (kind === 'video') { elements.video.error = { code: 3, message: 'bad H264' }; elements.video.emit('error'); }
      if (kind === 'sourcebuffer') env.buffers[0].emit('error');
      if (kind === 'sourceclose') { env.mediaSources[0].readyState = 'closed'; env.mediaSources[0].emit('sourceclose'); }
    }
    await task;
    assert.equal(player.session, null, kind);
    assert.equal(elements.status.dataset.state, 'error', kind);
    assert.match(elements.status.textContent, /stream ended|connection lost|decoding failed|fragmented MP4|closed unexpectedly|autoplay blocked/);
    assert.equal(env.reader.cancelCalls, 1, kind);
    assert.equal(env.intervals.size, 0, kind);
    assert.equal(env.fetchCalls.length, 1, 'do not automatically reconnect');
    assert.equal(elements.start.disabled, false);
  }
});

test('manual Start after an ended stream creates exactly one new playback session', async (t) => {
  const { env, elements, player } = harness({ chunks: [{ done: true }] });
  t.after(() => player.stop());
  await player.start();
  const oldReader = env.reader;
  assert.equal(oldReader.cancelCalls, 1);
  assert.equal(env.fetchCalls.length, 1);
  env.reader = new MockReader([{ value: new Uint8Array([1]), done: false }]);
  const task = player.start();
  assert.equal(player.start(), task);
  await until(() => env.reader.pending);
  assert.equal(env.fetchCalls.length, 2);
  assert.equal(env.fetchCalls[1].init.signal.aborted, false);
  assert.equal(elements.status.textContent, 'Playing live.');
  assert.equal(elements.start.disabled, true);
  player.stop();
  await task;
  assert.equal(env.reader.cancelCalls, 1);
  assert.equal(oldReader.cancelCalls, 1);
});

test('stop cancels a busy append as well as its updateend listeners', async (t) => {
  const { env, player } = harness({ autoComplete: false });
  t.after(() => player.stop());
  const task = player.start();
  await until(() => env.buffers[0]?.updating);
  player.stop();
  await task;
  assert.equal(env.buffers[0].abortCalls, 1);
  assert.equal(env.buffers[0].listenerCount, 0);
  assert.equal(env.reader.cancelCalls, 1);
});

test('stop cancels sourceopen waiting and a late fetch cannot reset a restarted player', async (t) => {
  const unopened = harness({ open: false });
  t.after(() => unopened.player.stop());
  const opening = unopened.player.start();
  await until(() => unopened.env.mediaSources.length === 1);
  unopened.player.stop();
  await opening;
  assert.equal(unopened.env.mediaSources[0].listenerCount, 0);
  assert.equal(unopened.env.reader.cancelCalls, 1);

  const deferred = [];
  const { env, elements, player } = harness({ fetch: (response) => new Promise((resolve) => deferred.push(() => resolve(response))) });
  t.after(() => player.stop());
  const oldTask = player.start();
  player.stop();
  const newTask = player.start();
  const current = player.session;
  deferred[0]();
  await oldTask;
  assert.equal(env.bodyCancelCalls, 1);
  assert.equal(player.session, current);
  assert.match(elements.status.textContent, /Connecting/);
  player.stop();
  deferred[1]();
  await newTask;
});

test('pagehide and unload cancel fetch; mount/destroy also clean their own UI listeners', async () => {
  for (const event of ['pagehide', 'beforeunload']) {
    const { env, elements } = harness();
    const mounted = mount(env.document, env);
    assert.ok(mounted);
    const task = mounted.session.done;
    await until(() => env.reader.pending);
    env.emit(event);
    await task;
    assert.equal(env.fetchCalls[0].init.signal.aborted, true);
    assert.equal(env.reader.cancelCalls, 1);
    assert.equal(elements.video.src, '');
    mounted.destroy();
    for (const target of [env, env.document, elements.start, elements.stop, elements.video]) assert.equal(target.listenerCount, 0);
  }
});
