import json
import re
from pathlib import Path
from collections import defaultdict

# -------------------------------------------------------------------------
INPUT_FILE  = Path("dev_Dataset/dev_tables.json")   # your Spider format file
OUTPUT_FILE = Path("ddl_schemas.json")
INDENT      = "    "                 # 4-space indent in CREATE TABLEs
TYPE_MAP = {                         # Spider → SQL types
    "text":    "TEXT",
    "integer": "INTEGER",
    "real":    "REAL",
    "date":    "DATE",
    "time":    "TIME",
    "boolean": "BOOLEAN",
}

# ---------- helper functions ---------------------------------------------
slug = lambda s: re.sub(r"\W+", "_", s.strip()).lower() or "col"

def build_tables(sch):
    """Return per-table structures ready for DDL rendering."""
    tbls = {
        i: {"name": slug(sch["table_names"][i]),
            "cols": [], "pk": [], "fk": defaultdict(list)}
        for i in range(len(sch["table_names"]))
    }

    # columns & types ------------------------------------------------------
    for g, (t, cname) in enumerate(sch["column_names"]):
        if t == -1:        # skip the synthetic '*'
            continue
        tbls[t]["cols"].append(
            (slug(cname), TYPE_MAP.get(sch["column_types"][g], "TEXT"), g)
        )

    # primary keys ---------------------------------------------------------
    for pk in sch["primary_keys"]:
        for g in (pk if isinstance(pk, list) else [pk]):
            t, _ = sch["column_names"][g]
            tbls[t]["pk"].append(g)

    # foreign keys ---------------------------------------------------------
    for local, ref in sch.get("foreign_keys", []):
        lt, lc = sch["column_names"][local]
        rt, rc = sch["column_names"][ref]
        tbls[lt]["fk"][rt].append((local, ref))

    return tbls


def ddl_for_schema(sch) -> str:
    """Generate one huge CREATE-TABLE string for a single DB."""
    parts = [f"-- {sch['db_id']}"]
    tbls = build_tables(sch)

    for t in tbls.values():
        lines = [f"CREATE TABLE {t['name']} ("]
        # columns
        col_lines = []
        for cname, ctype, gidx in t["cols"]:
            null = "NOT NULL" if gidx in t["pk"] else ""
            col_lines.append(f"{INDENT}{cname} {ctype} {null}".rstrip())
        # primary key
        if t["pk"]:
            pk_cols = ", ".join(slug(sch["column_names"][i][1]) for i in t["pk"])
            col_lines.append(f"{INDENT}PRIMARY KEY ({pk_cols})")
        # foreign keys
        for ref_tbl_idx, pairs in t["fk"].items():
            lcols = ", ".join(slug(sch["column_names"][l][1]) for l, _ in pairs)
            rcols = ", ".join(slug(sch["column_names"][r][1]) for _, r in pairs)
            ref_tbl = slug(sch["table_names"][ref_tbl_idx])
            col_lines.append(f"{INDENT}FOREIGN KEY ({lcols}) REFERENCES {ref_tbl} ({rcols})")
        lines.append(",\n".join(col_lines))
        lines.append(");")
        parts.append("\n".join(lines))

    return "\n\n".join(parts)


# ---------- main ----------------------------------------------------------
def main():
    schemas = json.loads(Path(INPUT_FILE).read_text(encoding="utf-8"))
    out = [
        {"db_id": sch["db_id"], "db_schema": ddl_for_schema(sch)}
        for sch in schemas
    ]
    OUTPUT_FILE.write_text(
        json.dumps(out, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"✓  Wrote {OUTPUT_FILE}")

if __name__ == "__main__":
    main()