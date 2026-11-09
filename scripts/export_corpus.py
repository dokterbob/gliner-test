"""Export prose chunks from a localdb store into data/corpus.jsonl (read-only).

Only `text` and `table` blocks are exported; code, headings and frontmatter carry
no useful NER signal. Each line: {chunk_id, store, uri, title, block_kind,
heading_path, text}.
"""

import argparse
import json
import sqlite3
from pathlib import Path

DEFAULT_DB = Path.home() / "Library/Application Support/localdb/data/localdb.db"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--store", default="localdb")
    ap.add_argument("--kinds", default="text,table")
    ap.add_argument("--out", type=Path, default=Path("data/corpus.jsonl"))
    args = ap.parse_args()

    kinds = args.kinds.split(",")
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    q = f"""
        SELECT c.id, s.name, r.uri, r.title, c.block_kind, c.heading_path, c.text
        FROM chunks c
        JOIN stores s ON s.id = c.store_id
        JOIN resources r ON r.store_id = c.store_id AND r.id = c.resource_id
        WHERE s.name = ? AND c.block_kind IN ({",".join("?" * len(kinds))})
        ORDER BY r.uri, c.block_seq, c.seq_in_block
    """
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w") as f:
        for cid, store, uri, title, kind, hp, text in con.execute(q, [args.store, *kinds]):
            f.write(json.dumps({"chunk_id": cid, "store": store, "uri": uri, "title": title,
                                "block_kind": kind, "heading_path": hp, "text": text},
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} chunks to {args.out}")


if __name__ == "__main__":
    main()
