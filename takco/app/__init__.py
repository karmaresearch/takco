import sys, os, glob, csv, json, html, urllib, itertools, collections
from flask import Flask, escape, request, render_template, g, redirect
from pathlib import Path
import logging as log

import evaluate.dataset
from .data import load_kb
from .stats import get_kbgold, get_kbinfo, get_novelty, novelty_add_pct

# Run Flask: env PYTHONPATH=src/ FLASK_ENV=development FLASK_APP=app/app.py DATADIR=data/ flask run --port=5555
app = Flask(__name__, static_folder="_static")
app.jinja_env.filters["any"] = any
app.jinja_env.filters["all"] = all
app.jinja_env.filters["lookup"] = lambda ks, d: [d.get(k) for k in ks]

if os.environ.get("config"):
    import toml

    config = toml.load(Path(os.environ.get("config")).open())
else:
    config = dict(os.environ)

kbs = {k.get("name", k.get("class")): k for k in config.get("kbs", {})}


def get_datasets():
    if "datasets" not in g:
        datadir = Path(config["datadir"])
        datasets_file = datadir / Path("datasets-cache.json")
        if not os.path.exists(datasets_file) or not config.get("cache"):

            log.info("Re-loading datasets...")
            datasets = {}
            if "assets" in config:
                resourcedir = config.get("resourcedir")
                assets = config.get("assets", ())
                assets = {a.get("name", a.get("class")): a for a in assets}
                for name, params in assets.items():
                    dataset = evaluate.dataset.load(resourcedir, datadir, **params)
                    datasets[name] = list(dataset.tables)

            with open(datasets_file, "w") as fw:
                json.dump(datasets, fw)
        g.datasets = json.load(open(datasets_file))
    return g.datasets


def save_datasets():
    datadir = config.get("datadir")
    datasets_file = os.path.join(datadir, "datasets-cache.json")
    with open(datasets_file, "w") as fw:
        json.dump(g.datasets, fw)


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
                print(f"kb has {len(g.kb)} facts")
            except Exception as e:
                print(f"Could not load kb {kbs[kbid]} due to {e}")
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
    for datasetname, data in datasets.items():
        stats[datasetname] = dict(
            numtables=len(data),
            numrows=sum(1 for d in data if d["rows"] for _ in d["rows"]),
            numcols=sum(1 for d in data if d["rows"] for _ in zip(*d["rows"])),
            numents=sum(
                1
                for d in data
                if d["entities"]
                for res in d["entities"].values()
                for es in res.values()
                for _ in es
            ),
            numclass=sum(1 for d in data if d["classes"]),
            numprops=sum(
                1
                for d in data
                if d["properties"]
                for cps in d["properties"].values()
                for _ in cps.values()
            ),
        )

    return render_template(
        "home.html",
        tablenames={
            name: sorted(t["name"] for t in tables) for name, tables in datasets.items()
        },
        stats=stats,
        **get_kbs(),
    )


@app.route("/table")
def table():
    dataset = request.args.get("dataset", None, type=str)
    tablename = request.args.get("table", None, type=str)
    cache = request.args.get("cache", True, type=bool)
    tables = {t["name"]: t for t in get_datasets()[dataset]}
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
            kblinks, kbinfo = get_kbgold(kb, table)
            table.setdefault("kblinks", {}).setdefault(kb.name, kblinks)
            save_datasets()
        else:
            kbinfo = get_kbinfo(kb, table)

        print(f"kbinfo is about {len(kbinfo)} nodes")
        novelty = get_novelty(kb, table, kblinks)
    else:
        print(f"No KB")

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
    cache = request.args.get("cache", False, type=bool)

    tables = {t["name"]: t for t in get_datasets()[dataset]}
    tables = dict(sorted(tables.items()))
    for table in tables.values():
        table["numRows"] = len(table["rows"])
        table["numCols"] = len(table["rows"][0])

    table_novelty = {}
    kb = get_kb()
    if kb:
        kbinfo = {}
        for i, (name, table) in enumerate(tables.items()):
            kblinks = table.get("kblinks", {}).get(kb.name)
            if not cache or not kblinks:
                print(f"{i:4d} / {len(tables)} {name}")
                kblinks, kbinfo_ = get_kbgold(kb, table)
                kbinfo.update(kbinfo_)
                table.setdefault("kblinks", {}).setdefault(kb.name, kblinks)
            table_novelty[name] = table["kblinks"][kb.name]["novelty"]

        save_datasets()
    else:
        log.info(f"No kb")

    novelty = {}
    for kind in ["lbl", "cls", "prop"]:
        novelty[kind] = sum(n[kind] for n in table_novelty.values())
        novelty[kind + "_nomatch"] = sum(
            n.get(kind + "_nomatch", 0) for n in table_novelty.values()
        )
        novelty[kind + "_redundant"] = sum(
            n.get(kind + "_redundant", 0) for n in table_novelty.values()
        )
        novelty[kind + "_total"] = sum(
            n[kind + "_total"] for n in table_novelty.values()
        )
    novelty_add_pct(novelty)

    return render_template(
        "dataset.html",
        dataset=dataset,
        tables=tables,
        table_novelty=table_novelty,
        novelty=novelty,
        **get_kbs(),
    )
