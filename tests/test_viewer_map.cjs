const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

// The overview map of viewer/index.html (the M key), run from the page's own source without WebGL, a browser, a
// network or game data. The streaming section runs with it, so the chunk states the map draws are the real ones.
const html = readFileSync(join(__dirname, '..', 'viewer', 'index.html'), 'utf8');
const viewerModule = html.match(/<script type="module">([\s\S]*?)<\/script>/)[1].replace(/^import .*;$/gm, '');
const begin = html.indexOf('// ---------- chunk streaming ----------');
const mapBegin = html.indexOf('// ---------- overview map ----------', begin);
const end = html.indexOf('// ---------- main loop ----------', mapBegin);
assert(begin >= 0 && mapBegin > begin && end > mapBegin, 'viewer streaming and overview map sections must exist');
const source = html.slice(begin, end);
const settle = () => new Promise(setImmediate);

const chunk = (id, minX, minZ, maxX, maxZ, top = 50, extra = {}) =>
    ({ id, url: `${id}.glb`, bounds: [minX, 0, minZ, maxX, top, maxZ], ...extra });

// A canvas whose 2D context records every call with the colours in force, the panel around it and the info line.
function page(width, height) {
    const calls = [], listeners = new Map();
    const ctx = { fillStyle: '', strokeStyle: '', lineWidth: 0, font: '' };
    for (const op of ['setTransform', 'clearRect', 'beginPath', 'rect', 'fill', 'arc', 'stroke', 'moveTo', 'lineTo',
        'closePath', 'fillRect', 'fillText']) {
        ctx[op] = (...args) => calls.push({ op, args, fill: ctx.fillStyle, stroke: ctx.strokeStyle });
    }
    const canvas = {
        clientWidth: width, clientHeight: height, width: 300, height: 150,
        getContext: kind => { assert.equal(kind, '2d'); return ctx; },
        addEventListener: (type, handler, options) => listeners.set(type, { handler, options }),
    };
    const panel = {
        hidden: true,
        classList: { toggle(name, on) { assert.equal(name, 'hidden'); panel.hidden = on; } },
        querySelector: selector => { assert.equal(selector, 'canvas'); return canvas; },
    };
    const info = { textContent: '' };
    const document = {
        pointerLockElement: null, exits: 0,
        exitPointerLock() { this.exits++; this.pointerLockElement = null; },
        getElementById: id => id === 'map' ? panel : id === 'map-info' ? info : {},
        addEventListener() {},
    };
    return { calls, listeners, canvas, panel, info, document };
}

function fixture({ chunks = [], spawn = [0, 0, 0], camera = [0, 0, 0], width = 420, height = 420, dpr = 1,
    ...manifest } = {}) {
    const dom = page(width, height), requests = [];
    let now = 0;
    const context = vm.createContext({
        performance: { now: () => now },
        setTimeout: () => 1, clearTimeout() {},
        THREE: { MeshStandardMaterial: class {}, Group: class { add() {} }, DoubleSide: 2,
            LoaderUtils: { extractUrlBase: url => url.slice(0, url.lastIndexOf('/') + 1) || './' } },
        AbortController,
        fetch: (url, { signal }) => new Promise((resolve, reject) => {
            signal.addEventListener('abort', () => reject(signal.reason), { once: true });
            requests.push({ url, reject, load: () => resolve({ ok: true, url, arrayBuffer: async () => url }) });
        }),
        GLTFLoader: class { parseAsync() { return Promise.resolve({ scene: { traverse() {} } }); } },
        scene: { add() {}, remove() {} },
        camera: { position: { x: camera[0], y: camera[1], z: camera[2] } },
        world: { test: false, spawn, chunks, ...manifest },
        document: dom.document, devicePixelRatio: dpr, yaw: 0,
    });
    vm.runInContext(source, context);
    vm.runInContext(`globalThis.viewer = { toggleMap, refreshMap, mapTransform, stream, LOAD_R, MAP_COLORS,
        get zoom() { return mapZoom; }, get center() { return mapCenter; } }`, context);
    const event = type => (...args) => dom.listeners.get(type).handler(...args);
    return {
        ...dom, context, requests, viewer: context.viewer, position: context.camera.position,
        at(time) { now = time; },
        // The calls of the latest redraw.
        frame() { return dom.calls.slice(dom.calls.findLastIndex(call => call.op === 'setTransform')); },
        ops(op) { return this.frame().filter(call => call.op === op); },
        // World metres under a canvas pixel, with the transform the page itself reports.
        metres(offsetX, offsetY) {
            const t = context.viewer.mapTransform();
            return [(offsetX - t.x) / t.scale, (offsetY - t.y) / t.scale];
        },
        move: event('mousemove'), leave: event('mouseleave'), click: event('click'), wheel: event('wheel'),
    };
}
const near = (actual, expected, message) => {
    assert.equal(actual.length, expected.length, message);
    actual.forEach((value, i) => assert.ok(Math.abs(value - expected[i]) < 1e-9, `${message || ''} ${actual} is not ${expected}`));
};

test('the page has the map panel, and its legend shows the colours the map draws', () => {
    const panel = html.match(/<div id="map" class="hidden">([\s\S]*?)<div id="start">/);
    assert.ok(panel, 'the map panel must come before #start and start hidden');
    assert.match(panel[1], /<canvas role="img" aria-label="[^"]+"><\/canvas>/);
    assert.match(panel[1], /<div id="map-info"><\/div>/);
    assert.match(html.match(/<div id="help">([\s\S]*?)<\/div>/)[1], /<kbd>M<\/kbd> overview map/);
    // Above the full-page "Click to fly" layer, or a click on the map would lock the mouse instead.
    assert.match(html, /#map \{[^}]*z-index: 1;/);
    const { MAP_COLORS } = fixture().viewer;
    for (const [state, label] of [['loaded', 'loaded'], ['loading', 'loading'], ['failed', 'failed'], ['idle', 'not loaded']]) {
        assert.ok(panel[1].includes(`<i style="color:${MAP_COLORS[state]}">■</i> ${label}<`), `legend entry for ${state}`);
    }
});

test('M opens and closes the map once per press and leaves modified shortcuts to the browser', async () => {
    const dom = page(420, 420), handlers = new Map();
    const manifest = { spawn: [5, 6, 7], chunks: [chunk('a', 0, 0, 10, 10)] };
    await vm.runInNewContext(`(async () => { ${viewerModule} })()`, {
        THREE: {
            WebGLRenderer: class { domElement = { addEventListener() {} }; setPixelRatio() {} setSize() {} setAnimationLoop() {} },
            Scene: class { add() {} },
            PerspectiveCamera: class { position = { x: 0, y: 0, z: 0, fromArray([x, y, z]) { Object.assign(this, { x, y, z }); } }; },
            DirectionalLight: class { position = { set() {} }; },
            Color: class {}, Fog: class {}, HemisphereLight: class {}, Clock: class {}, MeshStandardMaterial: class {},
            Group: class { add() {} }, Vector3: class {},
        },
        document: { ...dom.document, body: { prepend() {} } },
        addEventListener: (type, handler) => handlers.set(type, handler),
        devicePixelRatio: 1, innerWidth: 800, innerHeight: 600, URLSearchParams, location: { search: '', hash: '' },
        performance: { now: () => 0 }, AbortController, setTimeout: () => 1, clearTimeout() {},
        // Readable both ways, so the test does not depend on how loadWorld() reads the body.
        fetch: async () => ({ ok: true, status: 200, json: async () => manifest, text: async () => JSON.stringify(manifest) }),
        GLTFLoader: class {},
    });
    const press = (modifiers = {}) => handlers.get('keydown')({ code: 'KeyM', ...modifiers });
    assert.equal(dom.panel.hidden, true);
    assert.equal(dom.calls.length, 0, 'nothing is drawn before the map is opened');
    press();
    assert.equal(dom.panel.hidden, false);
    assert.ok(dom.calls.some(call => call.op === 'rect'), 'opening draws the map at once');
    assert.equal(dom.info.textContent, '1 chunks · click: fly there · wheel: zoom');
    press({ repeat: true });
    assert.equal(dom.panel.hidden, false, 'a held key must not flicker the map');
    for (const modifier of ['ctrlKey', 'metaKey', 'altKey']) press({ [modifier]: true });
    assert.equal(dom.panel.hidden, false);
    press();
    assert.equal(dom.panel.hidden, true);
    assert.deepEqual([...dom.listeners.keys()].sort(), ['click', 'mouseleave', 'mousemove', 'wheel']);
    press();
    assert.equal(dom.panel.hidden, false);
    assert.equal(dom.listeners.size, 4, 'reopening must not add the listeners again');
});

test('opening the map releases a locked mouse, since the map is used with the cursor', () => {
    const map = fixture({ chunks: [chunk('a', 0, 0, 10, 10)] });
    map.document.pointerLockElement = {};
    map.viewer.toggleMap();
    assert.equal(map.document.exits, 1);
    map.viewer.toggleMap();
    map.viewer.toggleMap();
    assert.equal(map.document.exits, 1, 'no lock, nothing to release');
});

test('the whole world fits the canvas at one scale, centred, with a gap between neighbours', () => {
    // 4000 m by 1000 m in 420 by 220 pixels with a 10 pixel margin: 0.1 pixel per metre on both axes.
    const map = fixture({ chunks: [chunk('west', -1000, 0, 1000, 1000), chunk('east', 1000, 0, 3000, 1000)],
        width: 420, height: 220 });
    map.viewer.toggleMap();
    assert.deepEqual(map.ops('rect').map(call => call.args), [[10.5, 60.5, 199, 99], [210.5, 60.5, 199, 99]]);
    near(map.metres(210, 110), [1000, 500], 'the centre of the canvas is the centre of the world');
    assert.deepEqual(map.ops('clearRect')[0].args, [0, 0, 420, 220]);
});

test('a high-DPI screen gets a sharper canvas and the same drawing, up to twice the size', () => {
    for (const [dpr, pixels] of [[1, 1], [1.5, 1.5], [2, 2], [3, 2]]) {
        const map = fixture({ chunks: [chunk('a', 0, 0, 100, 100)], width: 400, height: 300, dpr });
        map.viewer.toggleMap();
        assert.deepEqual([map.canvas.width, map.canvas.height], [400 * pixels, 300 * pixels]);
        assert.deepEqual(map.ops('setTransform')[0].args, [pixels, 0, 0, pixels, 0, 0]);
        assert.deepEqual(map.ops('rect')[0].args, [60.5, 10.5, 279, 279]);
    }
});

test('each chunk is drawn in the colour of what the streaming code is doing with it', async () => {
    // Four chunks within the loading radius of the camera and one beyond it, west to east.
    const names = ['loaded', 'loading', 'failed', 'retrying', 'far'];
    const chunks = names.map((id, i) => chunk(id, (i === 4 ? 6 : i) * 1000, 0, (i === 4 ? 7 : i + 1) * 1000, 1000));
    const map = fixture({ chunks, camera: [2000, 0, 500] });
    const { MAP_COLORS } = map.viewer;
    const state = colour => Object.keys(MAP_COLORS).find(name => MAP_COLORS[name] === colour);
    const request = name => map.requests.findLast(candidate => candidate.url === `${name}.glb`);
    const fail = name => request(name).reject(new Error('synthetic failure'));
    map.viewer.toggleMap();
    const states = () => {
        map.viewer.toggleMap(); map.viewer.toggleMap();     // closing and opening redraws now
        const drawn = map.ops('rect').sort((a, b) => a.args[0] - b.args[0]);
        assert.equal(drawn.length, 5);
        return Object.fromEntries(drawn.map((call, i) => [names[i], state(call.fill)]));
    };
    const idle = { loaded: 'idle', loading: 'idle', failed: 'idle', retrying: 'idle', far: 'idle' };
    assert.deepEqual(states(), idle);

    map.viewer.stream();
    assert.deepEqual(map.requests.map(candidate => candidate.url).sort(), ['failed.glb', 'loaded.glb', 'loading.glb', 'retrying.glb']);
    assert.deepEqual(states(), { loaded: 'loading', loading: 'loading', failed: 'loading', retrying: 'loading', far: 'idle' });

    request('loaded').load(); fail('failed'); fail('retrying');
    await settle();
    // One failure is not the end: the chunk waits for its retry and is not marked.
    assert.deepEqual(states(), { ...idle, loaded: 'loaded', loading: 'loading' });

    // The third failure uses up the attempts of 'failed'; 'retrying' is still waiting for its second answer.
    for (const time of [1000, 3000]) {
        map.at(time);
        map.viewer.stream();
        fail('failed');
        await settle();
    }
    assert.equal(map.requests.filter(candidate => candidate.url === 'failed.glb').length, 3);
    assert.deepEqual(states(), { loaded: 'loaded', loading: 'loading', failed: 'failed', retrying: 'loading', far: 'idle' });
    assert.equal(new Set(Object.values(MAP_COLORS)).size, 4, 'four states, four colours');
});

test('the camera is an arrow along the view direction, with the loading radius around it and a ring at the spawn', () => {
    // 1000 m square in 420 pixels: 0.4 pixel per metre, world origin at pixel 10, 10.
    const map = fixture({ chunks: [chunk('a', 0, 0, 1000, 1000)], spawn: [250, 0, 750], camera: [500, 30, 500] });
    map.viewer.toggleMap();
    const [radius, spawn] = map.ops('arc');
    near(radius.args.slice(0, 3), [210, 210, map.viewer.LOAD_R * 0.4], 'loading radius');
    near(spawn.args.slice(0, 3), [110, 310, 4], 'spawn ring');
    // Yaw 0 looks down -z, which is up on the map; a quarter turn to the left looks down -x.
    near(map.ops('moveTo')[0].args, [210, 201], 'arrow tip, yaw 0');
    map.context.yaw = Math.PI / 2;
    map.viewer.toggleMap(); map.viewer.toggleMap();
    near(map.ops('moveTo')[0].args, [201, 210], 'arrow tip, quarter turn');
    assert.deepEqual(map.ops('lineTo').length, 2);
});

test('a camera outside the mapped world keeps its arrow at the edge of the map', () => {
    const map = fixture({ chunks: [chunk('a', 0, 0, 1000, 1000)], camera: [9000, 0, -9000] });
    map.viewer.toggleMap();
    near(map.ops('moveTo')[0].args, [412, 8 - 9], 'clamped to the top right corner, pointing up');
});

test('the scale bar is a round length of at most a quarter of the map', () => {
    for (const [size, label, pixels] of [[1000, '200 m', 80], [6144, '1 km', 400 / 6.144], [40, '10 m', 100], [30000, '5 km', 400 / 6]]) {
        const map = fixture({ chunks: [chunk('a', 0, 0, size, size)] });
        map.viewer.toggleMap();
        assert.equal(map.ops('fillText')[0].args[0], label);
        const [x, y, length, thickness] = map.ops('fillRect')[0].args;
        assert.deepEqual([x, y, thickness], [10, 408, 2]);
        assert.ok(Math.abs(length - pixels) < 1e-9 && length <= 105, `${size} m world: bar of ${length} pixels`);
    }
});

test('a placed world is marked with north and reports RAGE coordinates under the cursor', () => {
    const chunks = [chunk('c_0_0', 0, -1000, 1000, 0)];
    const placed = fixture({ chunks, layout: 'placements' }), grid = fixture({ chunks, layout: 'grid' });
    for (const map of [placed, grid]) {
        map.viewer.toggleMap();
        map.move({ offsetX: 110, offsetY: 310 });
    }
    assert.deepEqual(placed.ops('fillText').map(call => call.args[0]), ['200 m', 'N ↑']);
    assert.deepEqual(grid.ops('fillText').map(call => call.args[0]), ['200 m']);
    // glTF (x, z) = (250, -250) is RAGE (X, Y) = (250, 250): north is -z.
    assert.equal(placed.info.textContent, 'x 250, z -250 (RAGE X 250, Y 250) · c_0_0: not loaded');
    assert.equal(grid.info.textContent, 'x 250, z -250 · c_0_0: not loaded');
});

test('the info line follows the cursor and the chunk under it, and returns to the hint when the cursor leaves', async () => {
    const hostile = '<img src=x onerror=alert(1)>';
    const map = fixture({ chunks: [chunk(hostile, 0, 0, 500, 1000), chunk('east', 500, 0, 1000, 1000)] });
    map.viewer.toggleMap();
    assert.equal(map.info.textContent, '2 chunks · click: fly there · wheel: zoom');
    map.move({ offsetX: 110, offsetY: 210 });
    // Written as text: an id from world.json is never parsed as markup.
    assert.equal(map.info.textContent, `x 250, z 500 · ${hostile}: not loaded`);
    assert.deepEqual(Object.keys(map.info), ['textContent']);
    map.move({ offsetX: 2, offsetY: 2 });
    assert.equal(map.info.textContent, 'x -20, z -20');

    // A cursor that rests sees the chunk under it change state with the next redraw.
    map.move({ offsetX: 310, offsetY: 210 });
    assert.equal(map.info.textContent, 'x 750, z 500 · east: not loaded');
    map.viewer.stream();
    map.at(100); map.viewer.refreshMap();
    assert.equal(map.info.textContent, 'x 750, z 500 · east: loading');
    map.requests[1].load();
    await settle();
    map.at(200); map.viewer.refreshMap();
    assert.equal(map.info.textContent, 'x 750, z 500 · east: loaded');
    map.leave();
    assert.equal(map.info.textContent, '2 chunks · click: fly there · wheel: zoom');
});

test('a world with LOD sets draws grid cells and names the busiest tier of a cell', () => {
    const lod_bands = { hd: [0, 130], slod1: [130, 20000] }, cell = [0, 0, 256, 256];
    // The coarse tier's one model reaches far beyond its cell.
    const map = fixture({ lod_bands, camera: [128, 0, 128], chunks: [
        chunk('c_slod1_0_0', -4000, -4000, 4000, 4000, 900, { lod: 'slod1', cell }),
        chunk('c_hd_0_0', 10, 10, 200, 200, 40, { lod: 'hd', cell }),
        chunk('c_hd_1_0', 300, 10, 500, 200, 40, { lod: 'hd', cell: [256, 0, 512, 256] }),
    ] });
    map.viewer.toggleMap();
    // 512 m by 256 m in 400 by 400 pixels: 0.78125 pixel per metre, so a cell is 200 pixels.
    assert.deepEqual(map.ops('rect').map(call => call.args), [[10.5, 110.5, 199, 199], [10.5, 110.5, 199, 199], [210.5, 110.5, 199, 199]]);
    assert.equal(map.ops('arc').length, 1, 'no loading radius: each tier has its own band');
    map.viewer.stream();        // the camera stands in the first cell: its HD tier loads, SLOD1 does not
    map.move({ offsetX: 110, offsetY: 210 });
    assert.equal(map.info.textContent, 'x 128, z 128 · c_hd_0_0: loading (+1 more here)');
});

test('a cell the manifest check did not look at is not drawn: a chunk without a tier uses its bounds', () => {
    // manifestError() validates `cell` only together with `lod`, so this one can hold anything.
    const map = fixture({ chunks: [chunk('a', 0, 0, 1000, 1000, 50, { cell: 'not a cell' }),
        chunk('b', 1000, 0, 2000, 1000, 50, { cell: [NaN, 0, 1, 1] })], width: 420, height: 220 });
    map.viewer.toggleMap();
    assert.deepEqual(map.ops('rect').map(call => call.args), [[10.5, 10.5, 199, 199], [210.5, 10.5, 199, 199]]);
    map.move({ offsetX: 60, offsetY: 60 });
    assert.equal(map.info.textContent, 'x 250, z 250 · a: not loaded');
});

test('a click moves the camera there and lifts it above the full-detail geometry at that point', () => {
    const lod_bands = { hd: [0, 130], orphan_hd: [0, 1000], slod1: [130, 20000] }, cell = [0, 0, 1000, 1000];
    const map = fixture({ lod_bands, camera: [0, 20, 0], chunks: [
        chunk('hd', 0, 0, 500, 1000, 80, { lod: 'hd', cell }),
        chunk('orphan', 400, 0, 500, 1000, 300, { lod: 'orphan_hd', cell }),
        chunk('slod', 0, 0, 1000, 1000, 2000, { lod: 'slod1', cell }),
    ] });
    map.viewer.toggleMap();
    const fly = (offsetX, offsetY) => { map.click({ offsetX, offsetY }); return { ...map.position }; };
    assert.deepEqual(fly(90, 210), { x: 200, y: 90, z: 500 });      // 10 m above the HD chunk's top
    assert.deepEqual(fly(190, 50), { x: 450, y: 310, z: 100 });     // the tallest full-detail chunk there
    assert.deepEqual(fly(90, 210), { x: 200, y: 310, z: 500 });     // never down: the height already had stays
    map.position.y = 5;
    assert.deepEqual(fly(310, 210), { x: 750, y: 5, z: 500 });      // only the coarse tier there: height kept
    near(map.ops('moveTo')[0].args, [310, 201], 'the map shows the new position at once');
});

test('a chunk without a tier counts as full detail when the camera is lifted', () => {
    const map = fixture({ camera: [0, 0, 0], chunks: [chunk('a', 0, 0, 1000, 1000, 120)] });
    map.viewer.toggleMap();
    map.click({ offsetX: 210, offsetY: 210 });
    assert.deepEqual({ ...map.position }, { x: 500, y: 130, z: 500 });
});

test('the wheel zooms the map about the cursor, within limits, and does not reach the speed handler', () => {
    const map = fixture({ chunks: [chunk('a', 0, 0, 1000, 1000)] });
    map.viewer.toggleMap();
    assert.equal(map.listeners.get('wheel').options.passive, true);
    let stopped = 0;
    const wheel = (deltaY, deltaX = 0, offsetX = 110, offsetY = 310) =>
        map.wheel({ deltaX, deltaY, offsetX, offsetY, stopPropagation: () => stopped++ });
    const under = map.metres(110, 310);
    near(under, [250, 750]);
    wheel(-100);
    assert.equal(map.viewer.zoom, 1.25);
    near(map.metres(110, 310), under, 'the point under the cursor stays there');
    near(map.ops('rect')[0].args, [-14.5, -64.5, 499, 499], 'and the map is drawn zoomed at once');
    wheel(-100); wheel(-100);
    near(map.metres(110, 310), under, 'after three notches too');
    // A sideways scroll is a pan gesture, not a zoom; it still must not change the flying speed.
    const before = map.viewer.zoom;
    wheel(0, 120); wheel(10, 120);
    assert.equal(map.viewer.zoom, before);
    for (let i = 0; i < 40; i++) wheel(-100);
    assert.equal(map.viewer.zoom, 64);
    for (let i = 0; i < 40; i++) wheel(100, 0, 300, 20);
    assert.equal(map.viewer.zoom, 1);
    assert.equal(map.viewer.center, null, 'zoomed all the way out, the world is centred again');
    assert.deepEqual(map.ops('rect')[0].args, [10.5, 10.5, 399, 399]);
    assert.equal(stopped, 85, 'every wheel event over the map stops there');
});

test('a click on a zoomed map lands on the point under the cursor', () => {
    const map = fixture({ chunks: [chunk('a', 0, 0, 1000, 1000, -10)], camera: [0, 77, 0] });
    map.viewer.toggleMap();
    for (let i = 0; i < 6; i++) map.wheel({ deltaX: 0, deltaY: -100, offsetX: 330, offsetY: 90, stopPropagation() {} });
    const [x, z] = map.metres(200, 150);
    map.click({ offsetX: 200, offsetY: 150 });
    near([map.position.x, map.position.z], [x, z]);
    assert.equal(map.position.y, 77);
    assert.ok(x > 700 && x < 800 && z > 200 && z < 300, `zoomed in around the cursor: ${x}, ${z}`);
});

test('an open map redraws at most every 100 ms and a closed one not at all', () => {
    const map = fixture({ chunks: [chunk('a', 0, 0, 1000, 1000)] });
    const draws = () => map.calls.filter(call => call.op === 'clearRect').length;
    map.at(5000); map.viewer.refreshMap();
    assert.equal(draws(), 0, 'closed: the frame loop draws nothing');
    map.viewer.toggleMap();
    assert.equal(draws(), 1);
    for (const [time, expected] of [[5000, 1], [5099, 1], [5100, 2], [5150, 2], [5200, 3]]) {
        map.at(time); map.viewer.refreshMap();
        assert.equal(draws(), expected, `at ${time} ms`);
    }
    map.viewer.toggleMap();
    map.at(9000); map.viewer.refreshMap();
    assert.equal(draws(), 3);
});

test('a world without chunks shows the camera and the spawn around the spawn point', () => {
    const map = fixture({ spawn: [100, 20, -300], camera: [100, 20, -300] });
    map.viewer.toggleMap();
    assert.equal(map.ops('rect').length, 0);
    assert.equal(map.info.textContent, '0 chunks · click: fly there · wheel: zoom');
    near(map.metres(210, 210), [100, -300], '1024 m around the spawn');
    near(map.ops('arc')[1].args.slice(0, 3), [210, 210, 4]);
    map.click({ offsetX: 10, offsetY: 410 });
    assert.deepEqual({ ...map.position }, { x: -412, y: 20, z: 212 });
});

test('a hidden or collapsed map panel is not drawn into', () => {
    const map = fixture({ chunks: [chunk('a', 0, 0, 1000, 1000)], width: 0, height: 0 });
    map.viewer.toggleMap();
    assert.equal(map.calls.length, 0);
    assert.deepEqual([map.canvas.width, map.canvas.height], [300, 150]);
});
