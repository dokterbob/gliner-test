"""run_coreml.py with relation descriptions: relations that carry `description_short` are sent as
Schema().relations({prompt: description}) (experiment 04). Same CLI as scripts/run_coreml.py."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_coreml as rc  # noqa: E402

_orig = rc.load_schemas


def load_schemas(schema_path, Schema):
    cfg = json.loads(Path(schema_path).read_text())
    ent_labels = [e["label"] for e in cfg["entities"]]
    label_to_tag = {e["label"]: e["ontonotes"] for e in cfg["entities"]}
    ent_schema = Schema().entities(ent_labels)
    passes = []
    for pn in cfg["relation_passes"]:
        rels = [r for r in cfg["relations"] if r["pass"] == pn]
        p2n = {r["prompt"]: r["name"] for r in rels}
        spec = {r["prompt"]: (r.get("description_short") or "") for r in rels}
        passes.append((pn, Schema().relations(spec), p2n))
    return ent_schema, passes, label_to_tag, cfg.get("threshold", 0.5)


rc.load_schemas = load_schemas
rc.main()
