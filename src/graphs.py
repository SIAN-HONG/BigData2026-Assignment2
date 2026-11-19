"""Q1: schema graph (8 tables, FK -> PK edges) and relational entity graph for
cohort users 601395036021679 and 283459088190688 (anonymized codes) in [2026-09-15 00:00, 2026-09-17 00:00) UTC.

Entity graph construction (from the data, not hand-drawn):
  1. activity rows of the two users whose row-time column (first_seen_at for PRs / issues /
     reviews, created_at for comments / pushes) lies in the UTC interval; for PRs and
     issues the user can be author / merged_by / assignee, for the others the actor column;
  2. every entity those rows point to through a foreign key (1 hop: repos, PRs, issues,
     reviews, parent review comments, users);
  3. for PRs / issues added in step 2, their own repo and author FKs (so that e.g. the author
     of a reviewed PR is visible).  All FK edges between included rows are drawn.
Usage: python src/graphs.py   (needs graphviz `dot`; set GRAPHVIZ_DOT if not on PATH)
Also writes schema_graph_slide for the report: schema with PK / FK columns only and FK names on the
rows instead of the edges.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pandas as pd

from common import FIG, ROOT, load_tables

DOT = os.environ.get("GRAPHVIZ_DOT") or shutil.which("dot") or str(ROOT / ".gv" / "bin" / "dot")

# table -> (pk, [(fk column, referenced table)], row-time column) exactly as in SCHEMA.md
SCHEMA = {
    "Users": ("user_id", [], "first_seen_at"),
    "Repositories": ("repo_id", [], "first_seen_at"),
    "PullRequests": ("pr_id", [("repo_id", "Repositories"), ("author_id", "Users"),
                               ("merged_by_id", "Users")], "first_seen_at"),
    "Issues": ("issue_id", [("repo_id", "Repositories"), ("author_id", "Users"),
                            ("pull_request_id", "PullRequests"), ("assignee_id", "Users")], "first_seen_at"),
    "IssueComments": ("comment_id", [("issue_id", "Issues"), ("repo_id", "Repositories"),
                                     ("author_id", "Users")], "created_at"),
    "PRReviews": ("review_id", [("pr_id", "PullRequests"), ("repo_id", "Repositories"),
                                ("reviewer_id", "Users")], "first_seen_at"),
    "PRReviewComments": ("comment_id", [("pr_id", "PullRequests"), ("repo_id", "Repositories"),
                                        ("author_id", "Users"), ("pull_request_review_id", "PRReviews"),
                                        ("in_reply_to_id", "PRReviewComments")], "created_at"),
    "Pushes": ("push_id", [("user_id", "Users"), ("repo_id", "Repositories")], "created_at"),
}
COLOR = {"Users": "#cfe2ff", "Repositories": "#d1e7dd", "PullRequests": "#fff3cd", "Issues": "#f8d7da",
         "IssueComments": "#e2d9f3", "PRReviews": "#ffe5d0", "PRReviewComments": "#fde2f3", "Pushes": "#e9ecef"}
# figure styling only (no data): saturated header / border per table; edge colour = the referenced table
# (white text on every HEAD colour has contrast >= 4.5:1)
HEAD = {"Users": "#2f67ad", "Repositories": "#2e7d55", "PullRequests": "#8c6a00", "Issues": "#b84444",
        "IssueComments": "#7558ad", "PRReviews": "#b35a1f", "PRReviewComments": "#b0487f", "Pushes": "#66737a"}
EDGE = {"Users": "#2f6fbf", "Repositories": "#2e8b57", "PullRequests": "#b8860b", "Issues": "#c0392b",
        "IssueComments": "#7a5fb0", "PRReviews": "#d35400", "PRReviewComments": "#c2185b", "Pushes": "#666666"}
FONT = "DejaVu Sans"


def render(dot_src: str, stem: str):
    p = FIG / f"{stem}.dot"
    p.write_text(dot_src)
    for fmt in ("svg", "png", "pdf"):
        subprocess.run([DOT, f"-T{fmt}", "-Gdpi=200" if fmt == "png" else "-Gdpi=72", str(p),
                        "-o", str(FIG / f"{stem}.{fmt}")], check=True)
    print("wrote", FIG / f"{stem}.svg/.png/.pdf")


def schema_graph(T, slide=False):
    title = ("8 tables; each edge = foreign key (FK) -> referenced primary key (PK)" if slide else
             "Schema graph: 8 tables, edges = foreign key -> referenced primary key\\n"
             f"row counts: data/db, latest Pushes.created_at {T['Pushes'].created_at.max():%Y-%m-%d %H:%M} UTC")
    L = ['digraph Schema {', f'graph [rankdir=LR, fontname="{FONT}", nodesep=%s, ranksep=%s, pad=0.2,'
         ' label="%s", labelloc=t, fontsize=18];' % (("0.35", "1.3", title) if slide else ("0.5", "1.1", title)),
         f'node [shape=plaintext, fontname="{FONT}"%s]; edge [penwidth=1.6, arrowsize=0.8, arrowhead=vee];'
         % (", fontsize=15" if slide else "")]
    fk_cols = {t: {c for c, _ in fks} for t, (_, fks, _) in SCHEMA.items()}
    for t, (pk, fks, rt) in SCHEMA.items():
        cols = list(T[t].columns)
        rows = [f'<tr><td bgcolor="{HEAD[t]}" colspan="2" cellpadding="6"><font color="white"><b>{t}</b></font></td></tr>']
        for c in cols:
            tag = ""
            if c == pk:
                tag = "PK"
            elif c in fk_cols[t]:
                tag = "FK"
            if slide and not tag:
                continue   # slide version: PK and FK columns only
            if not tag and c not in (rt,) and not c.startswith("github_"):
                continue   # keep the figure readable: keys, GitHub id and row time only
            if c == rt and not slide:
                tag = (tag + " time").strip()
            if c.startswith("github_"):
                tag = "GitHub id"
            name = f"<b>{c}</b>" if tag.startswith("PK") else c
            tcol = HEAD[dict(fks)[c]] if tag.startswith("FK") else "#555555"
            bg = f' bgcolor="{COLOR[t]}"' if tag.startswith("PK") else ""
            tsize = "" if slide else ' point-size="9"'   # slide: tags as large as the column names
            # PK: arrows arrive at the left of the name; FK: arrows leave right of the tag (tag in the target's colour)
            p1, p2 = (f' port="{c}"', "") if tag.startswith("PK") else ("", f' port="{c}"')
            rows.append(f'<tr><td{p1}{bg} align="left">{name}</td><td{p2}{bg} align="right">'
                        f'<font color="{tcol}"{tsize}><b>{tag}</b></font></td></tr>')
        if not slide:
            rows.append(f'<tr><td colspan="2"><font point-size="9" color="#888888">{len(cols)} columns, {len(T[t]):,} rows</font></td></tr>')
        L.append(f'{t} [label=<<table border="1" color="{HEAD[t]}" cellborder="0" cellspacing="0" cellpadding="{4 if slide else 3}">{"".join(rows)}</table>>];')
    n = 0
    for t, (_, fks, _) in SCHEMA.items():
        for c, ref in fks:
            lab = "" if slide else f', label="{c}", fontsize=9, fontcolor="{EDGE[ref]}"'
            L.append(f'{t}:{c}:e -> {ref}:{SCHEMA[ref][0]}:w [color="{EDGE[ref]}"{lab}];')
            n += 1
    L.append("}")
    render("\n".join(L), "schema_graph_slide" if slide else "schema_graph")
    print("FK edges:", n)


def entity_graph(T, gids=(601395036021679, 283459088190688), start="2026-09-15 00:00", end="2026-09-17 00:00"):
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    u = T["Users"]
    seeds = u[u.github_user_id.isin(gids)].user_id.tolist()
    actor_cols = {"PullRequests": ["author_id", "merged_by_id"], "Issues": ["author_id", "assignee_id"],
                  "IssueComments": ["author_id"], "PRReviews": ["reviewer_id"],
                  "PRReviewComments": ["author_id"], "Pushes": ["user_id"]}
    nodes: dict[tuple[str, int], str] = {}   # (table, pk) -> role: seed / activity / context
    for uid in seeds:
        nodes[("Users", uid)] = "seed"
    for t, cols in actor_cols.items():
        d = T[t]
        rt = SCHEMA[t][2]
        m = d[cols].isin(seeds).any(axis=1) & (d[rt] >= a) & (d[rt] < b)
        for pk in d.loc[m, SCHEMA[t][0]]:
            nodes[(t, int(pk))] = "activity"

    def add_fk_targets(keys, tables=None):
        new = []
        for (t, pk) in keys:
            row = T[t].iloc[pk]
            for c, ref in SCHEMA[t][1]:
                if tables and ref not in tables:
                    continue
                v = row[c]
                if pd.notna(v) and (ref, int(v)) not in nodes:
                    nodes[(ref, int(v))] = "context"
                    new.append((ref, int(v)))
        return new

    hop1 = add_fk_targets([k for k, r in nodes.items() if r == "activity"])
    add_fk_targets([k for k in hop1 if k[0] in ("PullRequests", "Issues")], tables={"Repositories", "Users"})

    def label(t, pk):
        r = T[t].iloc[pk]
        if t == "Users":
            return f"User\\n{r.github_user_id}" + (" (cohort)" if r.is_cohort else "")
        if t == "Repositories":
            return f"Repository\\n{r['name']}"
        if t == "PullRequests":
            return f"PullRequest #{r.number}\\nopened {r.created_at:%m-%d %H:%M}" if pd.notna(r.created_at) else f"PullRequest #{r.number}"
        if t == "Issues":
            kind = "PR conversation" if r.is_pull_request else "Issue"
            return f"{kind} #{r.number}\\nopened {r.created_at:%m-%d %H:%M}"
        if t == "IssueComments":
            return f"IssueComment\\n{r.created_at:%m-%d %H:%M}"
        if t == "PRReviews":
            return f"PRReview ({r.state})\\n{r.submitted_at:%m-%d %H:%M}"
        if t == "PRReviewComments":
            return f"PRReviewComment\\n{r.file_extension} {r.created_at:%m-%d %H:%M}"
        if t == "Pushes":
            br = r.branch if len(r.branch) <= 20 else r.branch[:18] + ".."
            return f"Push\\n{br}\\n{r.created_at:%m-%d %H:%M}"

    nid = {k: f"{k[0]}_{k[1]}" for k in nodes}
    L = ['digraph Entities {', f'graph [rankdir=LR, fontname="{FONT}", nodesep=0.25, ranksep=0.9, overlap=false,'
         f' label="Relational entity graph: cohort users 601395036021679 and 283459088190688, {start} to {end} UTC\\n'
         'thick border = seed user, solid fill = activity row in the interval, dashed = referenced entity (via FK)",'
         ' labelloc=t, fontsize=16];',
         f'node [shape=box, style="rounded,filled", fontname="{FONT}", fontsize=10]; edge [fontname="{FONT}", fontsize=8, arrowsize=0.7, penwidth=1.2];']
    for k, role in nodes.items():
        t = k[0]
        style = "rounded,filled" + (",dashed" if role == "context" else "") + (",bold" if role == "seed" else "")
        pen = 3 if role == "seed" else 1
        L.append(f'{nid[k]} [label="{label(*k)}", fillcolor="{COLOR[t]}", color="{HEAD[t] if role != "seed" else "#1f3f6e"}", style="{style}", penwidth={pen}];')
    edges = 0
    for (t, pk) in nodes:
        row = T[t].iloc[pk]
        for c, ref in SCHEMA[t][1]:
            v = row[c]
            if pd.notna(v) and (ref, int(v)) in nodes:
                L.append(f'{nid[(t, pk)]} -> {nid[(ref, int(v))]} [label="{c}", color="{EDGE[ref]}", fontcolor="{EDGE[ref]}"];')
                edges += 1
    L.append("}")
    render("\n".join(L), "entity_graph")
    cnt = pd.Series([k[0] for k in nodes]).value_counts()
    print("entity graph nodes:", dict(cnt), "edges:", edges)
    pd.DataFrame([(t, pk, r) for (t, pk), r in nodes.items()], columns=["table", "pk", "role"]) \
        .to_csv(FIG / "entity_graph_nodes.csv", index=False)


def entity_zoom(pr_number=3022, stem="entity_graph_zoom"):
    """Detail of figures/entity_graph.dot around one PR: the PR, every row with an FK edge to it, and the
    rows those reference. Node labels and edges are copied from the full graph (no new data); the
    repo_id edges of the review and line-comment rows that point to the PR are left out to keep it readable."""
    import re
    src = (FIG / "entity_graph.dot").read_text().splitlines()
    node_line = {m.group(1): l for l in src if (m := re.match(r"^(\w+) \[label=", l))}
    edges = [(m.group(1), m.group(2), m.group(3)) for l in src
             if (m := re.match(r'^(\w+) -> (\w+) \[label="(\w+)"[^\]]*\];', l))]
    # locate the PR by its number in the full graph (row numbers change between snapshots); must be unique
    hits = [n for n, l in node_line.items() if n.startswith("PullRequests_") and f'#{pr_number}\\n' in l]
    if len(hits) != 1:
        raise SystemExit(f"entity_zoom: expected one PullRequest #{pr_number} in entity_graph.dot, found {hits}")
    focus = hits[0]
    into = {s for s, t, _ in edges if t == focus}
    keep = {focus} | into | {t for s, t, _ in edges if s in into | {focus}}
    kept = [(s, t, c) for s, t, c in edges if s in keep and t in keep and not (s in into and c == "repo_id")]
    L = ['digraph Zoom {', f'graph [rankdir=TB, fontname="{FONT}", nodesep=0.15, ranksep=0.35, margin=0];',
         f'node [shape=box, style="rounded,filled", fontname="{FONT}", fontsize=16, margin="0.06,0.03"]; '
         f'edge [fontsize=13, arrowsize=0.7, penwidth=1.4, fontname="{FONT}"];']
    L += [node_line[n] for n in node_line if n in keep]
    col = lambda n: EDGE[n.rsplit("_", 1)[0]]
    L += [f'{s} -> {t} [label="{c}", color="{col(t)}", fontcolor="{col(t)}"];' for s, t, c in kept]
    L.append("}")
    render("\n".join(L), stem)
    print("zoom nodes:", len(keep), "edges:", len(kept))


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--zoom"]:
        entity_zoom()
        raise SystemExit
    T = load_tables()
    schema_graph(T)
    entity_graph(T)
    entity_zoom()
    schema_graph(T, slide=True)
