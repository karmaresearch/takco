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
            if 'tableData' in t:
                docs.append(t)
        except Exception as e:
            log.warn(e)
    return docs


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

    def pipe(self, func, *args, **kwargs):
        it = self.wrap(self.it) if isinstance(self, HashBag) else self
        it = (copy.deepcopy(x) for x in it)
        log.debug(f"Piping {func.__name__} ...")
        return self.new(func(it, *args, **kwargs))

    def fold(self, key, binop, exe=None, cast=False, keep_key=False):
        it = self.wrap(self.it) if isinstance(self, HashBag) else self
        it = (copy.deepcopy(x) for x in it)

        combined = {}
        for i in it:
            k = key(i)
            if k in combined:
                combined[k] = binop(combined[k], i)
            else:
                combined[k] = i
        return self.new(combined.items() if keep_key else combined.values())

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
                fw = open(f, "w")
            for r in it:
                if r:
                    print(json.dumps(r), file=fw)
                    fw.flush()
                    yield r
            fw.close()

        return self.__class__(dumped_it(f))

    @classmethod
    def load(cls, files, **kwargs):
        from io import TextIOBase
        from pathlib import Path

        def it(files):
            if type(files) != list:
                files = [files]
            for f in files:
                try:
                    if isinstance(f, TextIOBase):
                        yield from robust_json_loads_lines(f)
                    elif f == "-":
                        log.debug(f"Loading json from standard input")
                        import sys

                        yield from cls.load(sys.stdin)
                    elif Path(f).exists() and not Path(f).is_dir():
                        log.debug(f"Opening {f}")
                        with Path(f).open() as o:
                            yield from cls.load(o)
                    elif "*" in str(f):
                        import glob

                        yield from cls.load(glob.glob(str(f)))
                    elif Path(f).exists() and Path(f).is_dir():
                        yield from cls.load(Path(f).glob("*.jsonl"))
                    else:
                        raise Exception(f"Could not load {f}!")
                except GeneratorExit:
                    return

        return cls(it(files))


try:
    import tqdm, inspect  # type: ignore

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


except:
    log.debug(f"Could not load tqdm")

try:
    import dask.bag as db
    import dask.diagnostics
    import sys

    class DaskHashBag(HashBag):
        """A HashBag that uses the `Dask <http://dask.org>`_ library."""

        def start_client(self, **kwargs):
            global client
            from dask.distributed import Client  # type: ignore

            try:
                self.client = Client(**kwargs)
            except Exception as e:
                log.warn(e)

        def __init__(self, it=(), npartitions=1, client=None, **kwargs):
            self.client = client
            self.kwargs = kwargs

            if kwargs:
                self.start_client(**kwargs)

            if isinstance(it, db.Bag):
                self.bag = it
            else:
                it = list(it)
                npartitions = npartitions or len(it)
                self.bag = db.from_sequence(it, npartitions=npartitions)

        def new(self, it):
            return DaskHashBag(it, npartitions=self.bag.npartitions, client=self.client)

        def __repr__(self):
            kwargs = {'npartitions':self.bag.npartitions, **self.kwargs}
            args = (f'{k}={v.__repr__()}' for k,v in kwargs.items())
            return f"DaskHashBag(%s)" % ', '.join(args)

        @classmethod
        def load(cls, f, **kwargs):
            from io import TextIOBase

            if isinstance(f, TextIOBase):
                return cls(robust_json_loads_lines(f), **kwargs)
            else:
                return cls(
                    db.read_text(f).map_partitions(robust_json_loads_lines), **kwargs
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

        def fold(self, key, binop, exe=None, keep_key=False):
            bag = self.bag.foldby(key, binop=binop)
            if not keep_key:
                bag = bag.map(lambda x: x[1])
            return self.new(bag.repartition(self.bag.npartitions))

        def offset(self, get_attr, set_attr, default=0):
            df = self.bag.to_dataframe(columns=[get_attr])
            log.info(f"Dask offset {get_attr} {set_attr} {default}")
            if default:
                df[get_attr] = default
            vs = df[get_attr].cumsum() - df[get_attr]

            def setval(x, v):
                return {set_attr: v, **x}

            return self.new(self.bag.map(setval, vs.to_bag()))

        def dump(self, f, **kwargs):
            from io import TextIOBase

            if isinstance(f, TextIOBase):
                HashBag.dump(self.bag.compute(), f)
                return self
            else:
                self.bag.map(json.dumps).to_textfiles(f, last_endline=True)
                return self.load(f)


except Exception as e:
    log.debug(f"Could not load Dask")
    log.debug(e)


def preview(tables, nrows=5, ntables=10, nchars=100, hide_correct_rows=False):
    """Show table previews in Jupyter"""
    import json
    from jinja2 import Environment, PackageLoader
    from IPython.display import HTML  # type: ignore

    if isinstance(tables, dict):
        tables = [tables]

    env = Environment(loader=PackageLoader("takco", "app"),)
    env.filters["any"] = any
    env.filters["all"] = all
    env.filters["lookup"] = lambda ks, d: [d.get(k) for k in ks]

    template = env.get_or_select_template("templates/onlytable.html")

    content = ""
    for i, table in enumerate(tables):
        table = copy.deepcopy(table)

        ri_ann = {}
        hidden_rows = {}
        if hide_correct_rows:
            for ci, res in table.get("gold", {}).get("entities", {}).items():
                for ri, es in res.items():
                    if es:
                        ri_ann[ri] = True

                        predents = table.get("entities", {}).get(ci, {}).get(ri, {})
                        hide = all(e in predents for e in es)
                        hidden_rows[ri] = hidden_rows.get(ri, True) and hide
                    else:
                        hidden_rows[ri] = hidden_rows.get(ri, True)

        if nrows and any(hidden_rows.values()):
            n, nshow = 0, 0
            for ri, h in sorted(hidden_rows.items()):
                n += 1
                nshow += int(not h)
                if nshow >= nrows:
                    break
        else:
            n = nrows

        def cellchars(s):
            return (s[:nchars] + " (...)") if len(s) > nchars else s

        table["tableData"] = table.get("tableData", [])
        rows = [
            [cellchars(c.get("text", "")) for c in r]
            for r in table.get("tableData", [])
        ]
        table["rows"] = rows[:n]
        headers = [[c.get("text") for c in r] for r in table.get("tableHeaders", [])]
        table["headers"] = headers

        table.setdefault("entities", {})
        table.setdefault("classes", {})
        table.setdefault("properties", {})

        t = template.render(
            table=json.loads(json.dumps(table)),
            annotated_rows=ri_ann,
            hidden_rows=hidden_rows,
        )
        more_rows = max(0, len(table.get("tableData", [])) - nrows) if nrows else 0
        if more_rows:
            t += f"<p>({more_rows} more rows)</p>"

        content += f"""
        <span style='align-self: flex-start; margin: 1em; '>
        {t}
        </span>
        """
        if ntables and i + 1 >= ntables:
            break

    return HTML(
        f"""
    <div style="width: 100%; overflow-x: scroll; white-space: nowrap; display:flex;">
    {content}
    </div>
    """
    )


def tableobj_to_dataframe(table):
    import pandas as pd  # type: ignore

    body = [[c.get("text", "") for c in r] for r in table.get("tableData", [])]
    head = [[c.get("text", "") for c in r] for r in table.get("tableHeaders", [])]
    if any(head):
        return pd.DataFrame(body, columns=pd.MultiIndex.from_tuples(list(zip(*head))))
    else:
        return pd.DataFrame(body)


def tableobj_to_html(table, nrows=None, uniq=None, number=False):
    tableData = table.get("tableData", [])
    if uniq:
        uniq_cols = set()
        for row in table.get("tableHeaders", []):
            for col, cell in enumerate(row):
                if cell.get("text").lower() == uniq.lower():
                    uniq_cols.add(col)
        if uniq_cols:
            tableData = []
            uniqvals = set()
            for row in table.get("tableData", []):
                for uc in uniq_cols:
                    if uc < len(row) and (row[uc].get("text", "") not in uniqvals):
                        tableData.append(row)
                        uniqvals.add(row[uc].get("text", ""))

    tableData = tableData[:nrows]

    body = [
        [c.get("tdHtmlString", f"<td>{c.get('text')}</td>") for c in r]
        for r in tableData
    ]
    body = [[c.replace("span=", "=") for c in row] for row in body]
    body_html = "".join(f'<tr>{"".join(row)}</tr>' for row in body)

    head = [
        [c.get("tdHtmlString", f"<th>{c.get('text')}</th>") for c in r]
        for r in table.get("tableHeaders", [])
    ]
    head = [[c.replace("span=", "=") for c in row] for row in head]
    if number and head:
        head = [[f"<th>{i}</th>" for i in range(len(head[0]))]] + head
    head_html = "".join(f'<tr>{"".join(row)}</tr>' for row in head)
    return "<table>" + head_html + body_html + "</table>"


def reform_dict(dictionary, t=tuple(), reform={}):
    for key, val in dictionary.items():
        t = t + (key,)
        if isinstance(val, dict) and all(isinstance(v, dict) for v in val.values()):
            reform_dict(val, t, reform)
        else:
            reform.update({t: val})
        t = t[:-1]
    return reform