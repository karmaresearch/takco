"""
tacko is a modular system for extracting knowledge from tables.
"""
__version__ = "0.1.0"

import typing
from pathlib import Path
import logging as log
import os, sys, defopt, json, toml, types, argparse

from . import *
from .util import *

config = {}


def load_tables(s):
    global config
    return TableSet.load(s, **config)


class SetConfig(argparse.Action):
    def __init__(self, option_strings, dest, nargs="?", **kwargs):
        super(SetConfig, self).__init__(option_strings, dest, nargs="?", **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        global config
        config = {}
        if self.dest and values:
            conf = Config(values)
            if conf:
                config.update(conf)
                log.info(f"Loaded config {conf}")
        else:
            config.update(os.environ)
            log.info(f"Loaded config from environment")

        for k, v in config.items():
            setattr(namespace, k, v)


class SetExecutor(argparse.Action):
    def __init__(self, option_strings, dest, nargs=1, **kwargs):
        super(SetExecutor, self).__init__(option_strings, dest, nargs=1, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        global config
        config = config or {}
        if self.dest and values:
            config["executor"] = values
            setattr(namespace, "executor", values[0])


class SetVerbosity(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(SetVerbosity, self).__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if self.dest:
            loglevel = getattr(log, self.dest.upper(), log.WARNING)
            log.getLogger().setLevel(loglevel)
        else:
            ll = os.environ.get("LOGLEVEL", "").upper()
            loglevel = getattr(log, ll, log.WARNING)
            log.getLogger().setLevel(loglevel)
            logfile = os.environ.get("LOGFILE", None)
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

    parser = defopt._create_parser(
        funcs,
        strict_kwonly=False,
        parsers={
            typing.Container[str]: str.split,
            typing.Dict: json.loads,
            typing.Any: lambda _: None,
            Config: Config,
            HashBag: HashBag,
            TableSet: load_tables,
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
        try:
            if isinstance(result, HashBag):
                if hasattr(args, "out") and args.out:
                    log.info(f"Writing {result} to {args.out}")
                    for line in result._dump(args.out):
                        pass
                else:
                    log.info(f"Writing {result} to stdout")
                    for line in result._dump(sys.stdout):
                        pass
            elif isinstance(result, (types.GeneratorType, map, filter)):
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
