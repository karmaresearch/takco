import json
import logging as log
import inspect
import copy
import typing
import functools

T = typing.TypeVar('T')

class Config(dict, typing.Generic[T]):
    """A wrapper for json or toml configuration hashes.
    Reads a file or a string.
    Resolves ``{name=x}`` hashes from kwargs.
    """

    def __init__(self, val, context=None):
        from pathlib import Path
        import logging as log
        import toml, json

        if not val:
            return

        context = {c.get("name", c.get("class")): Config(c) for c in (context or [])}

        if isinstance(val, dict):
            if context and ("resolve" in val):
                if val["resolve"] in context:
                    self.update(context[val.pop("resolve")])
                else:
                    raise Exception(
                        f'Error resolving config: cannot find {val["resolve"]} in {context}'
                    )
            self.update(
                {
                    k: Config(v, context=context.values()) if isinstance(v, dict) else v
                    for k, v in val.items()
                }
            )
        elif isinstance(val, str) and val in context:
            self["name"] = val
            self.update(context[val])
        elif isinstance(val, str) and val.endswith(".toml"):
            try:
                val = {
                    "name": Path(val).name.split(".")[0],
                    **toml.load(Path(val).open()),
                }
                self.__init__(val, context=context.values())
            except Exception as e:
                log.error(e)
                raise e
        else:
            config_parsers = {
                "json-file": lambda val: {
                    "name": Path(val).name.split(".")[0],
                    **json.load(Path(val).open()),
                },
                "toml-file": lambda val: {
                    "name": Path(val).name.split(".")[0],
                    **toml.load(Path(val).open()),
                },
                "json-string": json.loads,
                "toml-string": toml.loads,
            }
            for cpi, config_parse in config_parsers.items():
                try:
                    self.__init__(config_parse(val), **context)
                    break
                except Exception as e:
                    log.debug(f"Did not parse {val} with parser {cpi} due to error {e}")
            if not len(self):
                self["name"] = val
                log.info(f"Skipped config: {val} (context: {context})")

    def init_class(self, **context):
        if isinstance(self, dict) and "class" in self:
            self = Config(self)
            if inspect.isclass(context.get(self["class"])):
                cls = context[self.pop("class")]
                cls_params = inspect.signature(cls).parameters
                cls_has_kwargs = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in cls_params.values()
                )

                kwargs = {
                    k: Config.init_class(v, **context)
                    for k, v in self.items()
                    if (k in cls_params) or cls_has_kwargs
                }
                obj = cls(**kwargs)
                if "name" in self:
                    obj.name = self["name"]
                return obj
            else:
                raise Exception(f'CONFIG ERROR: Cannot find class "{self["class"]}" !')
        else:
            return self


class HashBag:
    """A flexible wrapper for computation streams."""

    def __init__(self, it, wrap=lambda x: x, **kwargs):
        self.it = it
        self.wrap = wrap

    def persist(self):
        self.it = list(self.wrap(self.it))

    def __iter__(self):
        return iter(self.it)

    def _pipe(self, func, *args, desc=None, **kwargs):
        it = self.wrap(self.it) if isinstance(self, HashBag) else self
        it = (copy.deepcopy(x) for x in it)
        if desc:
            log.info(desc)
        return self.__class__(func(it, *args, **kwargs))

    def _fold(self, key, combine, exe=None, cast=False):
        it = self.wrap(self.it) if isinstance(self, HashBag) else self
        combined = {}
        for i in it:
            k = key(i)
            if k in combined:
                combined[k] = combine(i, combined[k])
            else:
                combined[k] = i
        return self.__class__(combined.values())

    def _offset(self, get_attr, set_attr, default=0):
        it = self.wrap(self.it) if isinstance(self, HashBag) else self

        def offset_it(it, default):
            total = 0
            for i in it:
                t = total + i.get(get_attr, default)
                yield i
                i[set_attr] = total
                total = t
            log.info(f"Serial cumsum {get_attr} {set_attr} {default} -> {total}")

        return self.__class__(offset_it(it, default))

    def _dump(self, f):
        import sys
        from io import TextIOBase
        from pathlib import Path

        it = self.wrap(self.it) if isinstance(self, HashBag) else self

        def dumped_it(f):
            if isinstance(f, TextIOBase):
                fw = f
            else:
                if "*" in str(f):
                    f = Path(str(f).replace("*", "output.jsonl"))
                fw = open(f, "w")
            for r in it:
                if r:
                    print(json.dumps(r), file=fw)
                    fw.flush()
                    yield r
            fw.close()

        return self.__class__(dumped_it(f))

    @classmethod
    def _load(cls, files, **kwargs):
        from io import TextIOBase
        from pathlib import Path

        def it(files):
            if type(files) != list:
                files = [files]
            for f in files:
                if isinstance(f, TextIOBase):
                    for li, l in enumerate(f):
                        try:
                            yield json.loads(l)
                        except:
                            log.debug(f"Failed to read json in line {li} of {f}")
                elif f == "-":
                    log.debug(f"Loading json from standard input")
                    import sys

                    yield from cls._load(sys.stdin)
                elif Path(f).exists() and Path(f).is_file():
                    log.debug(f"Opening {f}")
                    with Path(f).open() as o:
                        yield from cls._load(o)
                elif "*" in str(f):
                    import glob

                    yield from cls._load(glob.glob(str(f)))
                elif Path(f).exists() and Path(f).is_dir():
                    yield from cls._load(Path(f).glob("*"))

        return cls(it(files))


try:
    import tqdm

    class TqdmHashBag(HashBag):
        """A HashBag that displays `tqdm <https://tqdm.github.io/>`_ progress bars."""

        def __init__(self, it, **kwargs):
            def wrap(it):
                return tqdm.tqdm(it, leave=False)

            super().__init__(it, wrap=wrap, **kwargs)


except:
    log.debug(f"Could not load tqdm")

try:
    import dask.bag as db
    import dask.diagnostics
    import sys

    class DaskHashBag(HashBag):
        """A HashBag that uses the `Dask <http://dask.org>`_ library."""

        @staticmethod
        def start_client(**kwargs):
            global client
            from dask.distributed import Client

            try:
                client = Client(**kwargs)
            except Exception as e:
                log.warn(e)

        def __init__(self, it, npartitions=None, **kwargs):
            if kwargs:
                self.start_client(**kwargs)

            if isinstance(it, db.Bag):
                self.bag = it
            else:
                it = list(it)
                self.bag = db.from_sequence(it, npartitions=npartitions)

        @classmethod
        def _load(cls, f, **kwargs):
            if kwargs:
                cls.start_client(**kwargs)

            from io import TextIOBase

            if isinstance(f, TextIOBase):
                cls(json.loads(line) for line in f)
            else:
                return cls(db.read_text(f).map(json.loads))

        def persist(self):
            self.bag.persist()

        def __iter__(self):
            return iter(self.bag.compute())

        def _pipe(self, func, *args, desc=None, **kwargs):
            @functools.wraps(func)
            def listify(x, *args, **kwargs):
                return list(func(x, *args, **kwargs))

            return DaskHashBag(self.bag.map_partitions(listify, *args, **kwargs))

        def _fold(self, key, combine, exe=None, cast=False):
            bag = self.bag.foldby(key, binop=combine).map(lambda x: x[1])
            return DaskHashBag(bag.repartition(self.bag.npartitions))

        def _offset(self, get_attr, set_attr, default=0):
            df = self.bag.to_dataframe(columns=[get_attr])
            log.info(f"Dask offset {get_attr} {set_attr} {default}")
            if default:
                df[get_attr] = default
            vs = df[get_attr].cumsum() - df[get_attr]

            def setval(x, v):
                return {set_attr: v, **x}

            return DaskHashBag(self.bag.map(setval, vs.to_bag()))

        def _dump(self, f, **kwargs):
            from io import TextIOBase

            if isinstance(f, TextIOBase):
                HashBag._dump(self.bag.compute(), f)
                return self
            else:
                self.bag.map(json.dumps).to_textfiles(f, last_endline=True)
                return self._load(f)


except Exception as e:
    log.debug(f"Could not load Dask")
    log.debug(e)


def pages_download(
    ent_abouturl: typing.Collection[typing.Tuple[str, str]], encoding=None
):
    """Download html pages from urls
    
    Args:
        ent_abouturl: A ``[(uri,url)]`` list of entitu URIs and page URLs
        encoding: Page encoding (use "guess" to use requests' apparent encoding)
    """
    import requests

    for e, url in ent_abouturl:
        result = requests.get(url)
        if encoding:
            if encoding == "guess":
                result.encoding = result.apparent_encoding
            else:
                result.encoding = encoding
        if result.status_code == 200:
            yield {
                "url": url,
                "about": e,
                "html": result.text,
            }


def pages_warc(fnames):
    """Yield html pages from WARC files"""
    from warcio.archiveiterator import ArchiveIterator

    for fname in fnames:
        with open(fname, "rb") as stream:
            for record in ArchiveIterator(stream):
                if record.rec_type == "response":
                    url = record.rec_headers.get_header("WARC-Target-URI")
                    e = None
                    if "?about=" in url:
                        url, e = url.rsplit("?about=", 1)

                    text = record.content_stream().read().decode()
                    yield {
                        "url": url,
                        "about": e,
                        "html": text,
                    }


def preview(tables, nrows=5, ntables=10, hide_correct_rows=False):
    """Show table previews in Jupyter"""
    import json
    from jinja2 import Environment, PackageLoader
    from IPython.display import HTML

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

        table["tableData"] = table.get("tableData", [])
        rows = [[c.get("text") for c in r] for r in table.get("tableData", [])]
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
    import pandas as pd

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

    body = [[c.get("tdHtmlString", "<td></td>") for c in r] for r in tableData]
    body = [[c.replace("span=", "=") for c in row] for row in body]
    body = "".join(f'<tr>{"".join(row)}</tr>' for row in body)

    head = [
        [c.get("tdHtmlString", "<th></th>") for c in r]
        for r in table.get("tableHeaders", [])
    ]
    head = [[c.replace("span=", "=") for c in row] for row in head]
    if number and head:
        head = [[f"<th>{i}</th>" for i in range(len(head[0]))]] + head
    head = "".join(f'<tr>{"".join(row)}</tr>' for row in head)
    return "<table>" + head + body + "</table>"
