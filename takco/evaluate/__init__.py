import logging as log

from . import score
from . import dataset
from . import triples
from . import novelty


task_flatten = {
    "entities": score.flatten_entity_annotations,
    "properties": score.flatten_property_annotations,
    "classes": score.flatten_class_annotations,
}


def table_score(tablepairs, keycol_only=False):
    for table, goldtable in tablepairs:
        _id = table["_id"]

        table["gold"] = {}
        table["score"] = {}
        for task, flatten in task_flatten.items():
            gold = goldtable.get(task, {})
            table["gold"][task] = gold

            if gold:
                pred = table.get(task, {})
                if keycol_only and pred.get(str(table.get("keycol"))):
                    keycol = str(table.get("keycol"))
                    pred = {keycol: pred.get(keycol)}

                preds = dict(flatten(pred))
                golds = dict(flatten(gold))

                if preds and golds:
                    task_scores = score.classification(golds, preds)
                    task_scores["predictions"] = len(preds)
                    table["score"][task] = task_scores
            else:
                log.debug(f"No {task} annotations for {_id}")

        yield table


def table_novelty(tables, searcher):
    from . import novelty

    with searcher:
        for table in tables:

            triples = table.get("triples", [])
            noveltyhashes = novelty.get_cell_noveltyhashes(triples, searcher)

            kbnovelty = {
                "hashes": noveltyhashes,
                "counts": novelty.count_noveltyhashes(noveltyhashes),
            }

            kbname = searcher.name
            table.setdefault("novelty", {}).setdefault(kbname, kbnovelty)

            yield table


def table_triples(tables, include_type=True):
    """Make triples based on table predictions"""
    for table in tables:
        table["triples"] = list(triples.yield_triples(table, include_type=True))

        yield table


def pr_plot(show_curves, title=None, ylim=[0, 1.0]):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(nrows=1, ncols=1)

    # Isolines of constant F1-score
    f_scores = np.linspace(0, 1, num=20)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = ax.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)

    table = []
    for name, data in show_curves.items():
        r_curve, p_curve = data["recall"], data["precision"]
        ax.step(r_curve[1:-1], p_curve[1:-1], where="post", label=name)

    ax.legend()
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(ylim)
    ax.set_xlim([0.0, 1.0])
    if title:
        ax.set_title(title)

    plt.close(fig)
    return fig
