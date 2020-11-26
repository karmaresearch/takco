try:
    from sklearn.metrics import classification_report, precision_recall_curve
    from sklearn.exceptions import UndefinedMetricWarning
    import pandas as pd

    def classification(gold, pred, any_ok=False):
        import warnings

        df = pd.DataFrame({"gold": gold, "pred": pred}).fillna(0).applymap(bool)
        if any_ok:
            levels = list(range(len(df.index.levels)))[:-1]
            df = df.groupby(level=levels).apply(
                lambda g: g.head(1) if not g.pred.any() else g[g.pred].head(1)
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            return classification_report(df.gold, df.pred, output_dict=True).get("True")

    def pr_curve(gold, pred):
        import warnings

        df = pd.DataFrame({"gold": gold, "pred": pred}).fillna(0)
        df.gold = df.gold.apply(bool)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            p, r, t = precision_recall_curve(df.gold, df.pred)
            return dict(precision=list(p), recall=list(r), thresholds=list(t))


except ImportError:

    def classification(gold, pred):
        raise Exception("Scikit-learn or pandas could not be loaded")


def flatten_entity_annotations(entity_annot, key=None):
    for ci, col in entity_annot.items():
        for ri, uri_score in col.items():
            for uri, score in uri_score.items():
                k = tuple((key or []) + [int(ci), int(ri), str(uri)])
                yield k, score


def flatten_property_annotations(property_annot, key=None):
    for fromci, toci_props in property_annot.items():
        for toci, uri_score in toci_props.items():
            for uri, score in uri_score.items():
                k = tuple((key or []) + [int(fromci), int(toci), str(uri)])
                yield k, score


def flatten_class_annotations(class_annot, key=None):
    for ci, uri_score in class_annot.items():
        for uri, score in uri_score.items():
            k = tuple((key or []) + [int(ci), str(uri)])
            yield k, score
