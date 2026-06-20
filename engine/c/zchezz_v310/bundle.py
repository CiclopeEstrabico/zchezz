#!/usr/bin/env python3
"""
bundle.py — Zchezz single-file HTML bundler
Usage:
    python3 bundle.py zchezz_wasm.html zchezz_wasm.js zchezz_wasm.wasm nnue_weights.bin
Output:
    zchezz_bundle.html   (works offline, double-click in browser)
"""

import sys, base64, os, re, json

def parse_version(folder):
    """
    Convert folder name to version string injected into HTML.
    zchezz_v163B → 1.63b
    zchezz_v170C → 1.70c
    zchezz_v200  → 2.00
    zchezz_v153  → 1.53
    anything else → folder name as-is
    """
    m = re.search(r'[vV](\d+)([A-Za-z]?)$', folder)
    if not m:
        return folder
    digits = m.group(1)          # '163', '200', '153'
    suffix = m.group(2).lower()  # 'b', 'c', ''
    if len(digits) >= 3:
        major, minor = digits[:-2], digits[-2:]
    elif len(digits) == 2:
        major, minor = digits[0], digits[1] + '0'
    else:
        major, minor = digits, '00'
    return f'{major}.{minor}{suffix}'

def b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')

def read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()

# Regex patterns — immune to indentation changes
_START_RE = re.compile(
    r'(?m)^[ \t]*// Fetch wasm \+ weights.*\n[ \t]*async function startEngine\(\)[^\n]*\n'
)
_END_RE = re.compile(r'(?m)^[ \t]*startEngine\(\);')


def main():
    if len(sys.argv) != 5:
        print("Usage: python3 bundle.py <html> <engine.js> <engine.wasm> <weights.bin>")
        sys.exit(1)

    html_path, js_path, wasm_path, bin_path = sys.argv[1:]

    for p in [html_path, js_path, wasm_path, bin_path]:
        if not os.path.exists(p):
            print(f"ERROR: file not found: {p}"); sys.exit(1)

    print(f"Reading {wasm_path}    ({os.path.getsize(wasm_path)//1024} KB)")
    print(f"Reading {bin_path}     ({os.path.getsize(bin_path)//1024} KB)")
    print(f"Reading {js_path}      ({os.path.getsize(js_path)//1024} KB)")

    wasm_b64    = b64(wasm_path)
    weights_b64 = b64(bin_path)
    engine_js   = read(js_path)
    engine_js_json = json.dumps(engine_js)

    bundled = f"""// Fetch wasm + weights, then start worker (Web Worker — off-main-thread)
async function startEngine() {{
  dbg('startEngine() fired (bundled+worker) — ' + location.href, 'dbg-info');
  try {{
    const wasmBytes    = _b64ToBuffer(_WASM_B64);
    const weightsBytes = _b64ToBuffer(_WEIGHTS_B64);
    dbg('buffers decoded — wasm=' + wasmBytes.byteLength + 'b weights=' + weightsBytes.byteLength + 'b', 'dbg-info');

    // Build a Worker from inline source code
    const workerCode = _ENGINE_JS + '\\n\\n' + `
let Mod, bPtr, ready=false;
let _binit,_sinit,_bfen,_bapm,_nnld,_nnrst,_sbst;
self.onmessage=async function(e){{
  const msg=e.data;
  if(msg._init){{try{{await init(msg.wasmBytes,msg.weightsBytes);}}catch(err){{self.postMessage({{_status:'error',error:err.message}});}}return;}}
  if(!ready){{self.postMessage({{id:msg.id,error:'not ready'}});return;}}
  try{{self.postMessage(doSearch(msg.fen,msg.moves||'',msg.depth||9,msg.id,msg.timeLimitMs||0,msg.multiPv||1));}}catch(err){{self.postMessage({{id:msg.id,error:err.message}});}}
}};
async function init(wb,wts){{
  Mod=await ZchezzEngine({{noInitialRun:true,wasmBinary:wb,print:function(){{}},printErr:function(){{}}}});
  _binit=Mod.cwrap('board_init',null,[]);
  _sinit=Mod.cwrap('search_init',null,[]);
  _bfen=Mod.cwrap('board_load_fen',null,['number','number']);
  _bapm=Mod.cwrap('board_apply_moves',null,['number','number']);
  _nnld=Mod.cwrap('nnue_load_from_mem','number',['number','number']);
  _nnrst=Mod.cwrap('nnue_reset',null,[]);
  _sbst=Mod.cwrap('search_best_sret',null,['number','number','number']);
  _binit();_sinit();
  bPtr=Mod._malloc(524288);
  let ok=false;
  if(wts&&wts.byteLength>0){{
    const wp=Mod._malloc(wts.byteLength);Mod.HEAPU8.set(new Uint8Array(wts),wp);
    ok=(_nnld(wp,wts.byteLength)===0);Mod._free(wp);
  }}
  ready=true;self.postMessage({{_status:'ready',nnue:ok}});
}}
function hs(s){{const b=new TextEncoder().encode(s);const p=Mod._malloc(b.length+1);Mod.HEAPU8.set(b,p);Mod.HEAPU8[p+b.length]=0;return p;}}
function doSearch(fen,moves,depth,id,timeLimitMs,multiPv){{
  _binit();
  _nnrst();
  const fp=hs(fen);try{{_bfen(bPtr,fp);}}finally{{Mod._free(fp);}}
  if(moves&&moves.trim()){{
    const mp=hs(moves.trim());try{{_bapm(bPtr,mp);}}finally{{Mod._free(mp);}}
  }}
  const WASM_MAX_MS=10000;
  if(timeLimitMs<=0||timeLimitMs>WASM_MAX_MS) timeLimitMs=WASM_MAX_MS;
  const nPv=multiPv||1;
  const pp=Mod._malloc(32);
  Mod.HEAPU8.fill(0,pp,pp+32);
  const pdv=new DataView(Mod.HEAPU8.buffer,pp,32);
  pdv.setInt32(0,depth,true);
  pdv.setInt32(4,0,true);
  pdv.setInt32(8,timeLimitMs,true);
  pdv.setInt32(12,0,true);
  pdv.setInt32(16,nPv,true);
  pdv.setInt32(20,1,true);
  pdv.setInt32(24,0,true);
  pdv.setInt32(28,0,true);
  const SR_SZ=1920;
  const rp=Mod._malloc(SR_SZ);
  Mod.HEAPU8.fill(0,rp,rp+SR_SZ);
  try{{_sbst(rp,bPtr,pp);}}finally{{Mod._free(pp);}}
  const rv=new DataView(Mod.HEAPU8.buffer,rp,SR_SZ);
  const FL='abcdefgh',RK='87654321',PRC=['','','n','b','r','q'];
  function readMove(off){{const f=rv.getUint8(off),t=rv.getUint8(off+1),p=rv.getUint8(off+2);return{{f,t,p}};}}
  function moveStr(m){{if(m.f===0&&m.t===0)return'(none)';return FL[m.f&7]+RK[m.f>>3]+FL[m.t&7]+RK[m.t>>3]+(m.p>=2&&m.p<=5?PRC[m.p]:'');}}
  function readPv(off){{let s='';for(let i=0;i<240;i++){{const c=rv.getUint8(off+i);if(!c)break;s+=String.fromCharCode(c);}}return s;}}
  const best=readMove(0);
  const score=rv.getInt32(12,true);
  const pv=readPv(28);
  const isNull=(best.f===0&&best.t===0&&pv.length===0);
  const numPvs=rv.getInt32(284,true);
  let pvLines=null;
  if(nPv>1&&numPvs>0){{
    pvLines=[];
    for(let i=0;i<numPvs&&i<nPv;i++){{
      const mOff=1848+i*12;
      const mi=readMove(mOff);
      const si=rv.getInt32(288+i*4,true);
      const pi=readPv(312+i*256);
      const ui=moveStr(mi);
      if(ui!=='(none)')pvLines.push({{uci:ui,score:si,pv:pi}});
    }}
  }}
  Mod._free(rp);
  if(isNull)return{{id,uci:'(none)',score:0,nodes:0,pv:'',pvLines:null}};
  const uci=moveStr(best);
  return{{id,uci,score,nodes:0,pv,pvLines}};
}}
`;

    const worker = new Worker(URL.createObjectURL(new Blob([workerCode], {{ type: 'application/javascript' }})));
    worker.onerror = e => dbg('worker: ' + e.message, 'dbg-error');
    const cbs = {{}};
    worker.onmessage = e => {{
      const msg = e.data;
      if (msg._status) {{
        dbg('status: ' + JSON.stringify(msg), msg._status === 'error' ? 'dbg-error' : 'dbg-info');
        if (msg._status === 'ready') {{
          window.zchezzSearch = (p, cb) => {{ dbg('\\u2192 id=' + p.id + ' d=' + p.depth + ' mvs=' + ((p.moves || '').split(' ').length - 1), 'dbg-send'); cbs[p.id] = cb; worker.postMessage(p); }};
          if (window._onZchezzReady) window._onZchezzReady(msg.nnue);
        }}
        if (msg._status === 'error' && window._onZchezzReady) window._onZchezzReady(false);
        return;
      }}
      dbg('\\u2190 id=' + msg.id + ' uci=' + (msg.uci || '') + (msg.error ? ' ERR=' + msg.error : ''), 'dbg-recv');
      if (msg.id !== undefined && cbs[msg.id]) {{ cbs[msg.id](msg); delete cbs[msg.id]; }}
    }};
    worker.postMessage({{ _init: true, wasmBytes, weightsBytes }}, [wasmBytes, weightsBytes]);

  }} catch(err) {{
    dbg('startEngine ERROR: ' + err.message, 'dbg-error');
    if (window._onZchezzReady) window._onZchezzReady(false);
  }}
}}"""

    html = read(html_path)

    # Detect version from the current folder name and inject into HTML
    folder  = os.path.basename(os.path.abspath('.'))
    version = parse_version(folder)
    html    = html.replace('__VERSION__', version)
    print(f'[bundle] Version: {folder!r} -> {version!r}')

    # Helper to load SVG piece sets from pieces/ directories
    def load_svg_set(dirname):
        """Load SVG pieces from a directory with wK.svg, bK.svg, etc."""
        base = os.path.join(os.path.dirname(os.path.abspath(html_path)), '..', '..', '..', 'pieces', dirname)
        if not os.path.isdir(base):
            base = os.path.join(os.path.dirname(os.path.abspath('.')), 'pieces', dirname)
        if not os.path.isdir(base):
            return {}
        fmap = {'wK':'K','wQ':'Q','wR':'R','wB':'B','wN':'N','wP':'P',
                'bK':'k','bQ':'q','bR':'r','bB':'b','bN':'n','bP':'p'}
        result = {}
        for fname, key in fmap.items():
            path = os.path.join(base, fname + '.svg')
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    result[key] = f.read()
        return result

    # Extract Merida SVG pieces from files (real Merida, not python-chess CBurnett)
    svg_pieces = load_svg_set('merida')
    svg_json   = json.dumps(svg_pieces, ensure_ascii=False, separators=(',', ':'))
    html = html.replace(
        'const PIECE_SVG_DATA = {};  // __SVG_PIECES_PLACEHOLDER__',
        f'const PIECE_SVG_DATA = {svg_json};'
    )
    if svg_pieces:
        print(f'[bundle] Merida SVG injected into HTML ({len(svg_json)//1024} KB)')
    else:
        print('[bundle] No SVG pieces — Merida button will be hidden in bundled HTML')

    cb_pieces = load_svg_set('cburnett')
    cb_json = json.dumps(cb_pieces, ensure_ascii=False, separators=(',', ':'))
    html = html.replace(
        'const PIECE_SVG_CBURNETT = {};  // __SVG_CBURNETT_PLACEHOLDER__',
        f'const PIECE_SVG_CBURNETT = {cb_json};'
    )
    if cb_pieces:
        print(f'[bundle] CBurnett SVG injected ({len(cb_json)//1024} KB)')

    st_pieces = load_svg_set('staunty')
    st_json = json.dumps(st_pieces, ensure_ascii=False, separators=(',', ':'))
    html = html.replace(
        'const PIECE_SVG_STAUNTY = {};   // __SVG_STAUNTY_PLACEHOLDER__',
        f'const PIECE_SVG_STAUNTY = {st_json};'
    )

    m_start = _START_RE.search(html)
    if not m_start:
        print("ERROR: startEngine() marker not found in HTML.")
        print("Expected: // Fetch wasm + weights")
        print("followed: async function startEngine()")
        sys.exit(1)

    m_end = _END_RE.search(html, m_start.end())
    if not m_end:
        print("ERROR: standalone startEngine(); call not found.")
        sys.exit(1)

    constants = (
        f'const _WASM_B64    = "{wasm_b64}";\n'
        f'const _WEIGHTS_B64 = "{weights_b64}";\n'
        f'const _ENGINE_JS   = {engine_js_json};\n\n'
        'function _b64ToBuffer(b64str) {\n'
        '  const bin = atob(b64str);\n'
        '  const buf = new ArrayBuffer(bin.length);\n'
        '  const u8  = new Uint8Array(buf);\n'
        '  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);\n'
        '  return buf;\n'
        '}\n\n'
    )

    replacement = constants + bundled + "\n\nstartEngine();"
    html = html[:m_start.start()] + replacement + html[m_end.end():]

    out = 'zchezz_bundle.html'
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)

    kb = os.path.getsize(out) // 1024
    print(f"\nDone -> {out}  ({kb} KB / {kb/1024:.1f} MB)")
    print("Open zchezz_bundle.html directly in your browser — no server needed.")

if __name__ == '__main__':
    main()
