import sys, os, glob, csv, json, html, urllib, itertools, collections
from flask import Flask, escape, request, render_template, g, redirect
from pathlib import Path
import logging as log

from ..evaluate import dataset as eval_dataset
from .data import load_kb
from .stats import (
    get_kbgold,
    get_kbinfo,
)

# Run Flask: env PYTHONPATH=src/ FLASK_ENV=development FLASK_APP=app/app.py DATADIR=data/ flask run --port=5555
app = Flask(__name__, static_folder="_static")
app.jinja_env.filters["any"] = any
app.jinja_env.filters["all"] = all
app.jinja_env.filters["lookup"] = lambda ks, d: [d.get(k) for k in ks]

if os.environ.get("config"):
    import toml

    config = toml.load(Path(os.environ.get("config")).open())
else:
    config = {}
config.update( dict(os.environ) )

kbs = {k.get("name", k.get("class")): k for k in config.get("kbs", []) }
assets = {a.get("name", a.get("class")): a for a in config.get("assets", []) }

if os.environ.get("LOGLEVEL"):
    log.getLogger().setLevel(getattr(log, os.environ.get("LOGLEVEL").upper()))


def get_datasets():
    if "datasets" not in g:
        datadir = Path(config["datadir"])

        log.info("Re-loading datasets...")
        resourcedir = config.get("resourcedir")

        datasets = {}
        for datasetname, params in assets.items():
            dataset = eval_dataset.load(resourcedir, datadir, **params)
            datasets[datasetname] = {t["name"]: t for t in dataset.tables}
            
        g.datasets = datasets
    return g.datasets

def get_kbs():
    global kbs
    kbid = request.cookies.get("kbid")
    return {"kbid": kbid if kbid in kbs else None, "kbs": kbs}


def get_kb():
    global kbs
    if "kb" not in g:
        kbid = get_kbs().get("kbid")
        if kbid in get_kbs().get("kbs"):
            try:
                g.kb = load_kb(kbs[kbid])
                g.kb.name = kbid
                log.info(f"kb has {len(g.kb)} facts")
            except Exception as e:
                log.info(f"Could not load kb {kbs[kbid]} due to {e}")
                #                 raise e
                g.kb = None

        else:
            g.kb = None
    return g.kb


@app.route("/setcookies", methods=["POST"])
def setcookies():
    resp = redirect(request.referrer)
    for name in ["kbid"]:
        val = request.form.get(name, default=None, type=str)
        if val is not None:
            resp.set_cookie(name, val)
    return resp


@app.route("/")
def home():
    datasets = get_datasets()
    stats = {}
    for datasetname, tables in datasets.items():
        stats[datasetname] = dict(
            numtables=len(tables),
            numrows=sum(1 for _, d in tables.items() if d["rows"] for _ in d["rows"]),
            numcols=sum(
                1 for _, d in tables.items() if d["rows"] for _ in zip(*d["rows"])
            ),
            numents=sum(
                1
                for _, d in tables.items()
                if d["entities"]
                for res in d["entities"].values()
                for es in res.values()
                for _ in es
            ),
            numclass=sum(1 for _, d in tables.items() if d["classes"]),
            numprops=sum(
                1
                for _, d in tables.items()
                if d["properties"]
                for cps in d["properties"].values()
                for _ in cps.values()
            ),
        )

    return render_template(
        "home.html",
        tablenames={name: sorted(tables) for name, tables in datasets.items()},
        stats=stats,
        **get_kbs(),
    )


@app.route("/table")
def table():
    dataset = request.args.get("dataset", None, type=str)
    tablename = request.args.get("table", None, type=str)
    cache = request.args.get("cache", True, type=int)
    tables = get_datasets()[dataset]

    assert tablename in tables, f"{tablename} not in tables"
    table = tables[tablename]

    tablenames = sorted(tables)
    table_index = tablenames.index(tablename)
    prev_table = tablenames[table_index - 1]
    next_table = tablenames[table_index + 1 if table_index + 1 < len(tablenames) else 0]

    kbinfo, kblinks, novelty = None, None, None
    kb = get_kb()
    if kb:
        kblinks = table.get("kblinks", {}).get(kb.name)
        if not cache or not kblinks:
            log.info(f"Re-building kbinfo...")

            kblinks, kbinfo = get_kbgold(kb, table)
            table.setdefault("kblinks", {}).setdefault(kb.name, kblinks)
        else:
            kbinfo = get_kbinfo(kb, table)

        log.info(f"kbinfo is about {len(kbinfo)} nodes")
        novelty = table.get('novelty', {}).get(kb.name)
    else:
        log.info(f"No KB")

    annotated_rows = set()
    for ci, ri_ents in table.get("entities", {}).items():
        for ri, ents in ri_ents.items():
            annotated_rows.add(ri)

    return render_template(
        "table.html",
        dataset=dataset,
        tablename=tablename,
        tablenames=tablenames,
        prev_table=prev_table,
        table_index=table_index,
        next_table=next_table,
        table=table,
        annotated_rows=annotated_rows,
        kbinfo=kbinfo or {},
        kblinks=kblinks or {},
        novelty=novelty or {},
        **get_kbs(),
    )


@app.route("/dataset")
def dataset():
    dataset = request.args.get("dataset", None, type=str)
    cache = config.get("cache") or request.args.get("cache", True, type=int)

    tables = get_datasets()[dataset]
    tables = dict(sorted(tables.items()))
    for table in tables.values():
        table["numRows"] = len(table["rows"])
        table["numCols"] = len(table["rows"][0]) if table["numRows"] else 0

    novelty = {'counts': {}}
    kb = get_kb()
    if kb:
        for i, (name, table) in enumerate(tables.items()):
            task_counts = table.get("novelty", {}).get(kb.name, {}).get('counts', {})
            for task, counts in task_counts.items():
                novelty['counts'].setdefault(task, {})
                for c,v in counts.items():
                    novelty['counts'][task].setdefault(c, 0)
                    novelty['counts'][task][c] += v
    else:
        log.info(f"No kb")

    
    return render_template(
        "dataset.html",
        dataset=dataset,
        tables=tables,
        novelty=novelty,
        **get_kbs(),
    )
