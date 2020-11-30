import json
import os
import sys
import inspect
import argparse
import logging as log
import types
from pathlib import Path
from typing import Dict, Any

import toml
import defopt

import takco
from takco import TableSet, HashBag, TqdmHashBag
from . import config

assets: Dict[str, Any] = {}


class SetConfig(argparse.Action):
    def __init__(self, option_strings, dest, nargs="?", **kwargs):
        super(SetConfig, self).__init__(option_strings, dest, nargs="?", **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        global assets
        if self.dest and values:
            conf = config.parse(values)
            if isinstance(conf, dict) and conf:
                assets.update(conf)
                log.debug(f"Loaded config {conf}")
            else:
                raise Exception(f"Could not parse config {values}!")
        else:
            assets.update(os.environ)
            log.debug(f"Loaded config from environment")


class SetExecutor(argparse.Action):
    DEFAULT = TqdmHashBag()

    def __init__(self, option_strings, dest, nargs="?", **kwargs):
        super(SetExecutor, self).__init__(option_strings, dest, nargs="?", **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        global assets
        if self.dest:

            if values:
                executor = config.build(config.parse(values), assets)
            else:
                executor = self.DEFAULT

            assets["executor"] = executor
            setattr(namespace, "executor", executor)


class SetVerbosity(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(SetVerbosity, self).__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        logfmt = os.environ.get("LOGFMT", None)
        if logfmt:
            log.basicConfig(format=logfmt)

        if self.dest:
            loglevel = getattr(log, self.dest.upper(), log.WARNING)
            log.getLogger().setLevel(loglevel)
        else:
            ll = os.environ.get("LOGLEVEL", "").upper()
            loglevel = getattr(log, ll, log.WARNING)
            log.getLogger().setLevel(loglevel)
        logfile = os.environ.get("LOGFILE", None)
        if logfile:
            log.getLogger().addHandler(log.FileHandler(logfile))
        log.info(f"Set log level to {log.getLogger().getEffectiveLevel()}")


def main():

    funcs = (
        TableSet.run,
        TableSet.dataset,
        TableSet.extract,
        TableSet.reshape,
        TableSet.cluster,
        TableSet.integrate,
        TableSet.link,
        TableSet.coltypes,
        TableSet.score,
        TableSet.triples,
    )

    def getclasses(mod):
        for _, obj in inspect.getmembers(mod):
            if isinstance(obj, type):
                if obj.__module__.startswith(mod.__name__):
                    yield obj
            elif isinstance(obj, types.ModuleType):
                if hasattr(obj, "__name__") and obj.__name__.startswith(mod.__name__):
                    yield from getclasses(obj)

    def parse_tableset_arg(x):
        try:
            return TableSet.load(**config.build(config.parse(x), assets))
        except:
            return TableSet.load(x, executor=assets.get("executor"))

    parser = defopt._create_parser(
        funcs,
        strict_kwonly=False,
        parsers={
            Dict: config.parse,
            Any: lambda _: None,
            **{
                cls: lambda x: config.build(config.parse(x), assets)
                for cls in getclasses(takco)
            },
            HashBag: HashBag,
            TableSet: parse_tableset_arg,
        },
        argparse_kwargs={"description": __doc__},
    )
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for _, subparser in action.choices.items():
                subparser.add_argument(
                    "-C",
                    action=SetConfig,
                    metavar="CONFIG",
                    help="Use global configuration (see docs)",
                )
                subparser.add_argument(
                    "-X",
                    action=SetExecutor,
                    metavar="EXECUTOR",
                    help="Use executor (see docs)",
                )
                subparser.add_argument(
                    "-O", "--out", help="Write output to file(s)",
                )
                subparser.add_argument(
                    "-v", "--info", action=SetVerbosity, help="Log general information"
                )
                subparser.add_argument(
                    "-vv",
                    "--debug",
                    action=SetVerbosity,
                    help="Log debugging information",
                )

    args = parser.parse_args(sys.argv[1:])

    # Output result as json (or newline-delimited json if generator)
    result = defopt._call_function(parser, args._func, args)
    if result:
        log.info(f"End result is a {type(result)}")
        try:
            if isinstance(result, TableSet):
                result = result.tables

            if isinstance(result, HashBag):
                out = args.out if hasattr(args, "out") and args.out else sys.stdout
                log.info(f"Writing {result} to {out}")
                for _ in result.dump(out):
                    pass
            elif isinstance(result, (types.GeneratorType, map, filter)):  # type: ignore
                for r in result:
                    print(json.dumps(r))
            else:
                print(json.dumps(result))

        except IOError:
            log.debug(f"IOError")
            try:
                sys.stdout.close()
            except IOError:
                pass
    else:
        log.debug(f"No results")


if __name__ == "__main__":
    main()
