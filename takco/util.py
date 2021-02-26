import json
import logging as log
import inspect
import copy
import typing
import functools
import itertools

from .table import Table


def robust_json_loads_lines(lines):
    docs = []
    for line in lines:
        try:
            t = Table(json.loads(line))
            if "tableData" in t:
                docs.append(t)
        except Exception as e:
            log.warn(e)
    return docs

def json_dump(table):
    if isinstance(table, Table):
        table = table.to_dict()
    return json.dumps(table)

class HashBag:
    """A flexible wrapper for computation streams."""

    def __init__(self, it=(), wrap=lambda x: x, **kwargs):
        self.it = it
        self.wrap = wrap

    def new(self, it):
        return HashBag(it, wrap=self.wrap)

    @classmethod
    def concat(cls, hashbags):
        return hashbags[0].new(x for hb in hashbags for x in hb.it)

    def take(self, n):
        self.it = itertools.islice(self.it, n)
        return self

    def persist(self):
        self.it = list(self.wrap(self.it))
        return self

    def __iter__(self):
        for x in self.it:
            try:
                yield x
            except GeneratorExit:
                return

    def __len__(self):
        self.persist()
        return len(self.it)

    def pipe(self, func, *args, **kwargs):
        it = self.wrap(self.it) if isinstance(self, HashBag) else self
        it = (copy.deepcopy(x) for x in it)
        log.debug(f"Piping {func.__name__} ...")
        return self.new(func(it, *args, **kwargs))

    def fold(self, key, binop):
        it = self.wrap(self.it) if isinstance(self, HashBag) else self
        it = (copy.deepcopy(x) for x in it)

        combined = {}
        for i in it:
            k = key(i)
            if k in combined:
                combined[k] = binop(combined[k], i)
            else:
                combined[k] = i
        return self.new(combined.values())

    def fold_tree(self, key, binop):
        return self.fold(key, binop)

    def offset(self, get_attr, set_attr, default=0):
        it = self.wrap(self.it) if isinstance(self, HashBag) else self
        it = (copy.deepcopy(x) for x in it)

        def offset_it(it, default):
            total = 0
            for i in it:
                t = total + i.get(get_attr, default)
                i[set_attr] = total
                yield i
                total = t
            log.debug(f"Serial cumsum {get_attr} {set_attr} {default} -> {total}")

        return self.new(offset_it(it, default))

    def dump(self, f):
        import sys
        from io import TextIOBase
        from pathlib import Path

        it = self.wrap(self.it) if isinstance(self, HashBag) else self

        def dumped_it(f):
            if isinstance(f, TextIOBase):
                fw = f
            else:
                if "*.jsonl" in str(f):
                    f = Path(str(f).replace("*.jsonl", "output.jsonl"))
                log.debug(f"Writing to single file {f}")
                fw = open(f, "w")
            for r in it:
                if r:
                    print(json_dump(r), file=fw)
                    fw.flush()
                    yield r
            fw.close()

        if isinstance(f, str):
            Path(f).parent.mkdir(exist_ok=True, parents=True)

        return self.__class__(dumped_it(f))

    def load(self, *files):
        from io import TextIOBase
        from pathlib import Path

        def it(files):
            for f in files:
                if isinstance(f, list):
                    raise Exception(f"Cannot load HashBag from {f}")
                try:
                    if isinstance(f, TextIOBase):
                        yield from robust_json_loads_lines(f)
                    elif f == "-":
                        log.debug(f"Loading json from standard input")
                        import sys

                        yield from self.load(sys.stdin)
                    elif Path(f).exists() and not Path(f).is_dir():
                        log.debug(f"Opening {f}")
                        with Path(f).open() as o:
                            yield from self.load(o)
                    elif "*" in str(f):
                        import glob

                        yield from self.load(*glob.glob(str(f)))
                    elif Path(f).exists() and Path(f).is_dir():
                        yield from self.load(*Path(f).glob("*.jsonl"))
                    else:
                        raise Exception(f"Could not load {f}!")
                except GeneratorExit:
                    return

        return self.__class__(it(files))


import tqdm, inspect

class TqdmHashBag(HashBag):
    """A HashBag that displays `tqdm <https://tqdm.github.io/>`_ progress bars."""

    def __init__(self, it=(), **kwargs):
        def wrap(it):

            # Get calling function
            frameinfo = inspect.stack()[1]
            args = inspect.getargvalues(frameinfo.frame).locals
            relevant_arg = args.get("func", args.get("binop"))
            relevant_arg = relevant_arg.__name__ if relevant_arg else ""
            desc = f"{frameinfo.function}({relevant_arg})"

            return tqdm.tqdm(it, desc=desc, leave=False)

        super().__init__(it, wrap=wrap, **kwargs)


try:
    import dask.bag as db
    import dask.diagnostics
    import sys

    class DaskHashBag(HashBag):
        """A HashBag that uses the `Dask <http://dask.org>`_ library."""

        def start_client(self, **kwargs):
            global client
            from dask.distributed import Client

            try:
                client = Client(**kwargs)
                self.client = Client(**kwargs)
            except Exception as e:
                log.warn(e)

        def __init__(self, it=(), npartitions=None, client=None, **kwargs):
            self.client = client
            self.kwargs = kwargs
            self.try_npartitions = npartitions

            if kwargs:
                self.start_client(**kwargs)

            if isinstance(it, db.Bag):
                self.bag = it
            else:
                it = list(it)
                npartitions = npartitions or len(it) or None
                self.bag = db.from_sequence(it, npartitions=npartitions)

        def new(self, it):
            npartitions = max(self.try_npartitions or 1, self.bag.npartitions or 1)
            return DaskHashBag(it, npartitions=npartitions, client=self.client)

        def __repr__(self):
            kwargs = {"npartitions": self.bag.npartitions, **self.kwargs}
            args = (f"{k}={v.__repr__()}" for k, v in kwargs.items())
            return f"DaskHashBag(%s)" % ", ".join(args)

        def load(self, *f):
            cls = self.__class__
            from io import TextIOBase

            if isinstance(f, TextIOBase):
                return cls(robust_json_loads_lines(f), client=self.client)
            else:
                log.info(f"Reading {f} with {self.client}?")
                return cls(
                    db.read_text(f).map_partitions(robust_json_loads_lines),
                    client=self.client,
                )

        @classmethod
        def concat(cls, hashbags):
            return hashbags[0].new(db.concat([hb.bag for hb in hashbags]))

        def take(self, n):
            self.bag = self.bag.take(n, npartitions=-1, compute=False)
            return self

        def persist(self):
            try:
                self.bag = self.bag.persist()
            except Exception as e:
                log.error(e)
            return self

        def __iter__(self):
            return iter(self.bag.compute())

        def __len__(self):
            self.persist()
            return self.bag.count().compute()

        def pipe(self, func, *args, **kwargs):
            newargs = list(args)
            newkwargs = dict(kwargs)
            if self.client:
                try:
                    if newargs:
                        newargs = self.client.scatter(newargs, broadcast=True)
                    if newkwargs:
                        newkwargs = self.client.scatter(newkwargs, broadcast=True)
                except:
                    log.debug(f"Scattering for {func.__name__} failed!")

            @functools.wraps(func)
            def listify(x, *args, **kwargs):
                return list(func(x, *args, **kwargs))

            return self.new(self.bag.map_partitions(listify, *newargs, **newkwargs))

        def fold_tree(self, key, binop):
            return self.new(self.bag.foldby(key, binop=binop).map(lambda x: x[1]))

        def fold(self, key, binop):
            import pandas as pd

            def combine(df):
                return functools.reduce(binop, df.table)

            df = self.bag.map(lambda t: {'table':t} ).to_dataframe(meta={'table': 'object'})
            keymeta = pd.Series([key(t) for t in df.table.head(1)])
            index = df.table.apply(key, meta = keymeta)
            groups = df.assign(index=index).set_index("index").groupby("index")
            return self.new(groups.apply(combine).to_bag())

        def offset(self, get_attr, set_attr, default=0):

            d = self.bag.map(lambda t: {'table':t, get_attr: t.get(get_attr, default)})
            df = d.to_dataframe(meta={'table': 'object', get_attr: 'int'})
            vs = df[get_attr].cumsum() - df[get_attr]

            def setval(x, v):
                x[set_attr] = v
                return x

            return self.new(self.bag.map(setval, vs.to_bag()))

        def dump(self, f, **kwargs):
            from io import TextIOBase

            if isinstance(f, TextIOBase):
                HashBag.dump(self.bag.compute(), f)
                return self
            else:
                self.bag.map(json_dump).to_textfiles(f, last_endline=True)
                return self.load(f)


except Exception as e:
    log.debug(f"Could not load Dask")
    log.debug(e)



def reform_dict(dictionary, t=tuple(), reform=None):
    reform = reform or {}
    for key, val in dictionary.items():
        t = t + (key,)
        if isinstance(val, dict) and all(isinstance(v, dict) for v in val.values()):
            reform.update(reform_dict(val, t, reform))
        else:
            reform.update({t: val})
        t = t[:-1]
    return reform
