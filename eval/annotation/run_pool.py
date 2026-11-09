"""Pool driver: like run_coreml.py but splits texts that overflow the 128-subword bucket."""
import json, re, sys, time, gc
from pathlib import Path
sys.path.insert(0, "scripts")
import run_coreml as rc
from download_models import REVISION
from gliner_test.compiled_models import CACHE_ROOT

def split_text(text):
    mids=[m.end() for m in re.finditer(r"[.!?;:]\s+|\n", text)]
    if not mids: mids=[m.end() for m in re.finditer(r"\s+", text)]
    if not mids: return None
    c=len(text)/2
    cut=min(mids,key=lambda i:abs(i-c))
    if cut<=0 or cut>=len(text): return None
    return cut

def run(text, base, ctx, depth=0):
    try:
        e,r,l=rc.extract_window(*ctx, text, ctx_rel)
    except ValueError as ex:
        cut=split_text(text)
        if cut is None: raise
        a=run(text[:cut],base,ctx,depth+1); b=run(text[cut:],base+cut,ctx,depth+1)
        return [x+y for x,y in zip(a,b)]
    def sh(d): 
        d=dict(d); d["start"]+=base; d["end"]+=base; return d
    e=[sh(x) for x in e]
    r=[{**x,"head":sh(x["head"]),"tail":sh(x["tail"])} for x in r]
    return [e,r,[l]]

windows=[json.loads(l) for l in open(sys.argv[1])]
out=Path(sys.argv[2]); ctx_rel=float(sys.argv[3])
rt=rc.build_runtime(CACHE_ROOT/REVISION,"fp32","cpu")
from gliner2 import Schema
ent,passes,l2t,thr=rc.load_schemas(Path("schema/ontonotes.json"),Schema)
ctx=(rt,ent,passes,l2t,thr)
n=0;nsplit=0
with out.open("w") as f:
    for w in windows:
        e,r,l=run(w["text"],0,ctx)
        f.write(json.dumps({"window_id":w["window_id"],"entities":e,"relations":r,"latency_ms":sum(l),"chunks":len(l)},ensure_ascii=False)+"\n")
        nsplit+= len(l)>1
print("done",len(windows),"split",nsplit,file=sys.stderr)
