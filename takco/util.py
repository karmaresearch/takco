import json
import logging as log
import inspect


class Config(dict):
    """A wrapper for json or toml configuration hashes.
    Reads a file or a string.
    Resolves ``{name=x}`` hashes from kwargs.
    """

    def __init__(self, val, **context):
        from pathlib import Path
        import logging as log
        import toml, json

        if isinstance(val, dict):
            if context and ("name" in val) and (val["name"] in context):
                self["name"] = val["name"]
                self.update(context[val.pop("name")])
            self.update(
                {
                    k: Config(v, **context) if isinstance(v, dict) else v
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
                self.__init__(val, **context)
            except Exception as e:
                log.error(e)
                raise e
        else:
            config_parsers = [
                lambda val: {
                    "name": Path(val).name.split(".")[0],
                    **json.load(Path(val).open()),
                },
                lambda val: {
                    "name": Path(val).name.split(".")[0],
                    **toml.load(Path(val).open()),
                },
                json.loads,
                toml.loads,
            ]
            for cpi, config_parse in enumerate(config_parsers):
                try:
                    self.__init__(config_parse(val), **context)
                    break
                except Exception as e:
                    log.debug(f"Did not parse {val} with parser {cpi} due to error {e}")
            if not len(self):
                self["name"] = val
                log.info(f"Skipped config: {val} (context: {context})")

    def init_class(self, force=True, **context):
        if isinstance(self, dict) and "class" in self:
            self = dict(self)
            if inspect.isclass(context.get(self["class"])):
                c = context[self.pop("class")]
                kwargs = {
                    k: Config.init_class(v, force=False, **context)
                    for k, v in self.items()
                }
                return c(**kwargs)
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
        self.persist()
        return iter(self.it)

    def _pipe(self, func, *args, desc=None, **kwargs):
        it = self.wrap(self.it) if isinstance(self, HashBag) else self
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

    def _dump(self, f, force=False):
        import sys
        from io import TextIOBase
        from pathlib import Path

        it = self.wrap(self.it) if isinstance(self, HashBag) else self

        def dumped_it(f):
            if isinstance(f, TextIOBase):
                fw = f
            else:
                if "*" in str(f):
                    f = Path(str(f).replace("*", "0"))
                fw = open(f, "w")
            for r in it:
                if r:
                    print(json.dumps(r), file=fw)
                    yield r
            fw.close()

        if force:
            return self.__class__(list(dumped_it(f)))
        else:
            return self.__class__(dumped_it(f))

    @classmethod
    def _load(cls, files):
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
                    yield from cls._load(Path(f).open())
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
                return tqdm.tqdm(list(it), leave=False)

            super().__init__(it, wrap=wrap, **kwargs)


except:
    log.debug(f"Could not load tqdm")

try:
    import dask.bag as db
    import dask.diagnostics
    import sys

    #     pbar = dask.diagnostics.ProgressBar(out=sys.stderr)
    #     pbar.register()

    class DaskHashBag(HashBag):
        """A HashBag that uses the `Dask <http://dask.org>`_ library."""

        def __init__(self, it, npartitions=None, **kwargs):

            global cluster

            if isinstance(it, db.Bag):
                self.bag = it
            else:
                it = list(it)
                self.bag = db.from_sequence(it, npartitions=npartitions)

        def persist(self):
            self.bag.persist()

        def __iter__(self):
            return iter(self.bag.compute())

        def _pipe(self, func, *args, desc=None, **kwargs):
            def listified(x, *args, **kwargs):
                return list(func(x, *args, **kwargs))

            return DaskHashBag(self.bag.map_partitions(listified, *args, **kwargs))

        def _fold(self, key, combine, exe=None, cast=False):
            return DaskHashBag(self.bag.foldby(key, combine).map(lambda x: x[1]))

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

        @classmethod
        def _load(cls, f):
            from io import TextIOBase

            if isinstance(f, TextIOBase):
                cls(json.loads(line) for line in f)
            else:
                return cls(db.read_text(f).map(json.loads))


except Exception as e:
    log.debug(f"Could not load Dask")
    log.debug(e)


def get_warc_pages(fnames):
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


from collections.abc import Iterable
import json
import copy


def preview(tables, nrows=5, ntables=10):
    if isinstance(tables, dict):
        tables = [tables]

    from jinja2 import Environment, PackageLoader
    from IPython.display import HTML

    env = Environment(loader=PackageLoader("takco", "app"),)
    env.filters["any"] = any
    env.filters["all"] = all
    env.filters["lookup"] = lambda ks, d: [d.get(k) for k in ks]

    template = env.get_or_select_template("templates/onlytable.html")

    content = ""
    for i, table in enumerate(tables):
        table = copy.deepcopy(table)

        rows = [[c.get("text") for c in r] for r in table.get("tableData", [])][:nrows]
        table.setdefault("rows", rows)
        headers = [[c.get("text") for c in r] for r in table.get("tableHeaders", [])]
        table.setdefault("headers", headers)

        table.setdefault("entities", {})
        table.setdefault("classes", {})
        table.setdefault("properties", {})

        t = template.render(table=json.loads(json.dumps(table)))
        more_rows = max(0, len(table.get("tableData", [])) - nrows)
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
