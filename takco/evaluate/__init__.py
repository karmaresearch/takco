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


def table_score(tables, table_annot, keycol_only=False):
    for table in tables:
        _id = table["_id"]
        goldtable = table_annot.get(_id, {})

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


def table_novelty(tables, linker_config):
    from .. import link
    from . import novelty

    linker = linker_config.init_class(**link.__dict__)
    for table in tables:

        triples = table.get("triples", [])
        noveltyhashes = novelty.get_cell_noveltyhashes(triples, linker)

        kbnovelty = {
            "hashes": noveltyhashes,
            "counts": novelty.count_noveltyhashes(noveltyhashes),
        }

        kbname = linker.name
        table.setdefault("novelty", {}).setdefault(kbname, kbnovelty)

        yield table


def table_triples(tables, include_type=True):
    """Make triples based on table predictions"""
    for table in tables:
        table["triples"] = list(triples.yield_triples(table, include_type=True))

        yield table
