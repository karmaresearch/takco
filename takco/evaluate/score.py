import logging as log

try:
    from sklearn.metrics import classification_report, precision_recall_curve
    from sklearn.exceptions import UndefinedMetricWarning
    import pandas as pd

    def classification(gold, pred, any_annotated=False, only_annotated=False):
        import warnings

        df = pd.DataFrame({"gold": gold, "pred": pred}).fillna(0).applymap(bool)
        levels = list(range(len(df.index.levels)))[:-1]

        if only_annotated:
            # Not all targets are annotated, so only take annotated targets
            df = df[ df.groupby(level=levels).gold.transform('any') ]
    
        if any_annotated:
            # For each target (cell, column, column pair), there are multiple right answers
            # So, only take one per fn, fp and tp.
            anypred = df.pred.groupby(level=levels).transform('any')
            anycorrect = (df.pred & df.gold).groupby(level=levels).transform('any')

            df = pd.concat([
                df[~anypred].groupby(level=levels).head(1), # unpredicted (fn)
                df[(~anycorrect) & df.pred].groupby(level=levels).head(1), # incorrect (fp)
                df[df.pred & df.gold].groupby(level=levels).head(1), # correct (tp)
            ])

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
