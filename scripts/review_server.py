"""One-card-at-a-time review page for reports/review_queue.json (stdlib only).

Blind-then-reveal: the page gets the claim + photos; Terra/2f/Sol text, strata
and prices are served only from /api/reveal after (or on explicit request
before) the verdict. Verdicts are appended to a JSONL log (latest wins; undo =
a null verdict) so the session is resumable. Images are served by token from
an allowlist built at startup. Zero provider calls.

Run:
  .venv\\Scripts\\python.exe scripts\\review_server.py [--host 0.0.0.0] [--port 8765] [--rebuild]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import review_cards as rc  # noqa: E402

QUEUE = ROOT / "reports" / "review_queue.json"
VERDICTS = ROOT / "reports" / "review_verdicts.jsonl"


def image_tokens(cards):
    """{token: Path} for every photo path in the queue — the only files the server will send."""
    out = {}
    for c in cards:
        for s in c.get("strips") or []:
            for p in s.get("photos") or []:
                if p.get("path"):
                    out[hashlib.sha1(str(p["path"]).encode("utf-8")).hexdigest()[:16]] = Path(p["path"])
    return out


def public_cards(cards):
    pub = []
    for c in cards:
        b = rc.blind_card(c)
        for s in b["strips"]:
            for p in s["photos"]:
                p["url"] = "/img/" + hashlib.sha1(str(p.pop("path", "")).encode("utf-8")).hexdigest()[:16]
        pub.append(b)
    return pub


class State:
    def __init__(self, queue_path: Path, verdicts_path: Path):
        self.queue_path, self.verdicts_path = queue_path, verdicts_path
        q = json.loads(queue_path.read_text(encoding="utf-8"))
        self.cards = {c["card_id"]: c for c in q["cards"]}
        self.images = image_tokens(q["cards"])
        self.public = public_cards(q["cards"])
        self.lock = threading.Lock()

    def record(self, body: dict) -> dict:
        cid = body.get("card_id")
        if cid not in self.cards:
            raise KeyError(cid)
        rec = {"card_id": cid, "verdict": body.get("verdict"), "tag": body.get("tag") or None,
               "notes": (body.get("notes") or "").strip() or None, "peeked": bool(body.get("peeked")),
               "extra": body.get("extra") or None, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with self.lock:
            self.verdicts_path.parent.mkdir(parents=True, exist_ok=True)
            with self.verdicts_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec


def make_handler(state: State):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            p = urlparse(self.path).path
            if p == "/":
                return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if p == "/api/queue":
                return self._send(200, {"cards": state.public, "done": rc.latest_verdicts(state.verdicts_path)})
            if p.startswith("/api/reveal/"):
                card = state.cards.get(p.rsplit("/", 1)[1])
                return self._send(200, rc.reveal_payload(card)) if card else self._send(404, {"error": "no card"})
            if p.startswith("/img/"):
                path = state.images.get(p.rsplit("/", 1)[1])
                if not path or not path.is_file():
                    return self._send(404, {"error": "no image"})
                ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                data = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "max-age=86400")
                self.end_headers()
                return self.wfile.write(data)
            return self._send(404, {"error": "not found"})

        def do_HEAD(self):  # health probes
            self.send_response(200 if urlparse(self.path).path == "/" else 404)
            self.end_headers()

        def do_POST(self):
            p = urlparse(self.path).path
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, {"error": "bad json"})
            if p == "/api/verdict":
                try:
                    rec = state.record(body)
                except KeyError:
                    return self._send(404, {"error": "no card"})
                return self._send(200, {"ok": True, "record": rec, "reveal": rc.reveal_payload(state.cards[rec["card_id"]])})
            return self._send(404, {"error": "not found"})

        def log_message(self, fmt, *args):  # quieter log: only non-image, non-200 lines
            if "/img/" in self.path or (args and str(args[1]).startswith("200")):
                return
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    return H


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>v5 review</title>
<style>
body{font:15px/1.45 system-ui,Segoe UI,sans-serif;margin:0;background:#121212;color:#e8e8e8}
header{display:flex;gap:14px;align-items:center;padding:8px 14px;background:#1c1c1c;position:sticky;top:0;z-index:2;font-size:13px;color:#bbb;flex-wrap:wrap}
main{max-width:1100px;margin:0 auto;padding:10px 14px 60px}
h2{margin:8px 0 4px;font-size:18px;font-weight:600}
.claim{background:#1e1e1e;border-left:3px solid #5a9;padding:8px 12px;margin:8px 0}
.claim b{color:#9dd;font-size:16px}
.obs{margin:4px 0 0;padding-left:18px;color:#ccc}
.lbl{color:#999;font-size:12px;margin-top:10px;text-transform:uppercase;letter-spacing:.04em}
.strip{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0}
.strip img{max-width:100%;max-height:72vh;border:1px solid #333;background:#000}
.strip.muted img{opacity:.45;max-height:150px}
.strip img.lowres{outline:2px solid #b66}
.opts{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}
.opt{padding:6px 10px;border:1px solid #444;border-radius:6px;background:#1a1a1a;cursor:pointer;user-select:none}
.opt.sel{border-color:#6c6;background:#1f3a1f}
.opt.tag.sel{border-color:#cc6;background:#3a3a1f}
.key{color:#fc8;font-weight:600;margin-right:4px}
textarea{width:100%;box-sizing:border-box;min-height:46px;background:#1a1a1a;color:#eee;border:1px solid #444;border-radius:6px;padding:6px;font:inherit}
#reveal{display:none;background:#18202c;border:1px solid #345;border-radius:6px;padding:8px 12px;margin:10px 0}
#reveal.on{display:block}
.rv{margin:5px 0}.rv b{color:#9bd}
#status{color:#8d8}.hint{color:#888;font-size:12px}
#help{display:none;position:fixed;right:12px;bottom:12px;background:#222;border:1px solid #555;padding:10px 14px;border-radius:8px;font-size:13px;white-space:pre;z-index:3}
#help.on{display:block}
</style></head><body>
<header><span id="pos"></span><span id="src"></span><span id="addr"></span><span id="status"></span>
<span style="margin-left:auto" class="hint">? help</span></header>
<main>
 <h2 id="title"></h2>
 <div class="claim"><b id="cclaim"></b><ul class="obs" id="obs"></ul></div>
 <div id="strips"></div>
 <div class="lbl" id="vlbl">verdict</div><div class="opts" id="opts"></div>
 <div class="lbl" id="taglbl">error type (optional)</div><div class="opts" id="tags"></div>
 <textarea id="notes" placeholder="notes — n to focus, Esc to leave, Ctrl+Enter to save"></textarea>
 <div class="hint" id="subhint"></div>
 <div id="reveal"></div>
</main>
<div id="help">1-4  verdict (bathroom: digits = distinct bathrooms, p/o/u = billing)
a-e  error type (condition cards)        n  notes      Enter  save (then Enter/j = next)
j/k  next/prev   s  skip   z  undo       r  reveal early (recorded)   $  prices (after reveal)</div>
<script>
let Q=[],D={},i=0,sel={},peeked=false,showPrices=false,rev=null;
const $=id=>document.getElementById(id);
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function card(){return Q[i];}
async function load(){const j=await (await fetch('/api/queue')).json();Q=j.cards;D=j.done||{};
 i=Q.findIndex(c=>!D[c.card_id]);if(i<0)i=0;render();}
function render(){const c=card();sel={};peeked=false;showPrices=false;rev=null;$('reveal').className='';$('reveal').innerHTML='';
 if(!c){$('title').textContent='queue is empty';return;}
 const d=D[c.card_id],done=Object.keys(D).filter(k=>Q.some(q=>q.card_id===k)).length;
 const phaseN=Q.filter(q=>q.phase===c.phase).length,phaseDone=Q.filter(q=>q.phase===c.phase&&D[q.card_id]).length;
 $('pos').textContent=`phase ${c.phase} (${phaseDone}/${phaseN}) · card ${i+1}/${Q.length} · done ${done}`;
 $('src').textContent=`${c.source} · ${c.property_key} · ${c.run_id}`;$('addr').textContent=c.address||'';
 $('status').textContent=d?`✓ ${d.verdict}${d.tag?' ['+d.tag+']':''}${d.extra&&d.extra.distinct_bathrooms!=null?' baths='+d.extra.distinct_bathrooms:''}`:'';
 $('title').textContent=c.title;$('cclaim').textContent=c.claim.catalog_claim||'';
 $('obs').innerHTML=(c.claim.observations||[]).map(o=>`<li>${esc(o)}</li>`).join('');
 $('strips').innerHTML=c.strips.map(s=>`<div class="lbl">${esc(s.label)} (${s.photos.length})</div><div class="strip${s.muted?' muted':''}">`+
  s.photos.map(p=>{const t=p.wh?p.key+' ('+p.wh[0]+'×'+p.wh[1]+')':p.key;const lr=p.wh&&Math.min(p.wh[0],p.wh[1])<500;
   return `<a href="${p.url}" target="_blank"><img src="${p.url}"${lr?' class="lowres"':''} alt="${esc(t)}" title="${esc(t)}"></a>`;}).join('')+`</div>`).join('');
 const keys=Object.entries(c.verdict_keys||{});
 $('vlbl').textContent=c.kind==='bathroom'?'billing (digits set the distinct-bathroom count)':'verdict';
 $('opts').innerHTML=(c.kind==='bathroom'?`<span class="opt" id="bn">distinct bathrooms: <b id="bnv">?</b></span>`:'')+
  keys.map(([k,v])=>`<span class="opt" data-v="${v}"><span class="key">${k}</span>${v}</span>`).join('');
 const hasTags=c.kind==='condition';$('taglbl').style.display=hasTags?'':'none';$('tags').style.display=hasTags?'':'none';
 $('tags').innerHTML=Object.entries(c.tags||{}).map(([k,v])=>`<span class="opt tag" data-t="${v}"><span class="key">${k}</span>${v}</span>`).join('');
 $('notes').value=d&&d.notes||'';$('subhint').textContent='';
 if(d){sel={verdict:d.verdict,tag:d.tag,distinct_bathrooms:d.extra&&d.extra.distinct_bathrooms};paint();reveal();}
 window.scrollTo(0,0);}
function paint(){document.querySelectorAll('#opts .opt[data-v]').forEach(e=>e.classList.toggle('sel',e.dataset.v===sel.verdict));
 document.querySelectorAll('#tags .opt').forEach(e=>e.classList.toggle('sel',e.dataset.t===sel.tag));
 const b=$('bnv');if(b)b.textContent=sel.distinct_bathrooms==null?'?':sel.distinct_bathrooms;}
async function reveal(){const c=card();if(!c)return;rev=await (await fetch('/api/reveal/'+c.card_id)).json();paintReveal();}
function paintReveal(){if(!rev)return;const h=(rev.hidden&&showPrices)?`<div class="rv"><b>price</b> $${rev.hidden.low}/${rev.hidden.high}</div>`:(rev.hidden?`<div class="hint">$ shows prices</div>`:'');
 $('reveal').innerHTML=rev.reveal.map(r=>`<div class="rv"><b>${esc(r.label)}</b> ${esc(r.text)}</div>`).join('')+
  `<div class="rv"><b>strata</b> ${rev.strata.join(', ')}${rev.legacy_item_id?' · legacy '+esc(rev.legacy_item_id):''}</div>`+
  `<div class="rv hint">${esc(JSON.stringify(rev.meta))}</div>`+h;$('reveal').className='on';}
function nextUndone(){const j=Q.findIndex((c,k)=>k>i&&!D[c.card_id]);i=j>=0?j:Math.min(i+1,Q.length-1);render();}
function go(n){i=Math.max(0,Math.min(Q.length-1,i+n));render();}
async function submit(){const c=card();if(!c)return;
 if(!sel.verdict||(c.kind==='bathroom'&&sel.distinct_bathrooms==null)){$('subhint').textContent='pick a verdict first'+(c.kind==='bathroom'?' (and a bathroom count)':'');return;}
 const body={card_id:c.card_id,verdict:sel.verdict,tag:sel.tag||null,notes:$('notes').value,peeked:peeked,
  extra:c.kind==='bathroom'?{distinct_bathrooms:sel.distinct_bathrooms}:null};
 const j=await (await fetch('/api/verdict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
 if(j.ok){D[c.card_id]=j.record;rev=j.reveal;paintReveal();$('status').textContent='✓ saved — Enter or j for next';}}
async function undo(){const c=card();if(!c||!D[c.card_id])return;
 await fetch('/api/verdict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({card_id:c.card_id,verdict:null})});
 delete D[c.card_id];render();}
document.addEventListener('click',e=>{const o=e.target.closest('.opt');if(!o)return;
 if(o.dataset.v){sel.verdict=o.dataset.v;paint();}else if(o.dataset.t){sel.tag=sel.tag===o.dataset.t?null:o.dataset.t;paint();}});
document.addEventListener('keydown',e=>{const c=card();if(!c)return;const k=e.key;
 if(document.activeElement===$('notes')){if(k==='Escape')$('notes').blur();else if(k==='Enter'&&e.ctrlKey)submit();return;}
 if(e.ctrlKey||e.metaKey||e.altKey)return;
 if(k==='?'){$('help').classList.toggle('on');return;}
 if(k==='j'||k==='ArrowRight'||k==='s'){go(1);return;}
 if(k==='k'||k==='ArrowLeft'){go(-1);return;}
 if(k==='n'){e.preventDefault();$('notes').focus();return;}
 if(k==='z'){undo();return;}
 if(k==='r'){if(!rev){peeked=!D[c.card_id];reveal();}return;}
 if(k==='$'){showPrices=!showPrices;paintReveal();return;}
 if(k==='Enter'){if(D[c.card_id]&&rev)nextUndone();else submit();return;}
 if(c.verdict_keys&&c.verdict_keys[k]){sel.verdict=c.verdict_keys[k];paint();return;}
 if(c.kind==='condition'&&c.tags&&c.tags[k]){sel.tag=sel.tag===c.tags[k]?null:c.tags[k];paint();return;}
 if(c.kind==='bathroom'&&/^\d$/.test(k)){sel.distinct_bathrooms=+k;paint();return;}});
load();
</script></body></html>"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to review from a phone on the LAN")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--queue", type=Path, default=QUEUE)
    ap.add_argument("--verdicts", type=Path, default=VERDICTS)
    ap.add_argument("--rebuild", action="store_true", help="rebuild the queue (defaults) before serving")
    args = ap.parse_args(argv)
    if args.rebuild or not args.queue.is_file():
        args.queue.parent.mkdir(parents=True, exist_ok=True)
        args.queue.write_text(json.dumps(rc.build_queue(), indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"queue rebuilt: {args.queue}")
    state = State(args.queue, args.verdicts)
    done = rc.latest_verdicts(args.verdicts)
    print(f"{len(state.cards)} cards ({sum(1 for c in state.cards if c in done)} done), {len(state.images)} images; "
          f"verdicts -> {args.verdicts}")
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"serving http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}/  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
