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


def load_tables(s):
    global executor
    log.debug(f"Loading tables {s} using executor {executor}")
    return TableSet(executor._load(s))


class SetConfig(argparse.Action):
    def __init__(self, option_strings, dest, nargs="?", **kwargs):
        super(SetConfig, self).__init__(option_strings, dest, nargs="?", **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        config = {}
        if self.dest and values:
            conf = Config(values)
            if conf:
                config.update(conf)
                log.info(f"Loaded config {conf}")
        elif Path("config.toml").exists():
            config.update(toml.load(Path("config.toml").open()))
            log.info(f"Loaded local config.toml")
        else:
            config.update(os.environ)
            log.info(f"Loaded config from environment")

        for k, v in config.items():
            setattr(namespace, k, v)


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
        TableSet.score,
        TableSet.triples,
    )

    # hot-patch defopt to deal with hidden kwargs (VAR_KEYWORD)
    def signature(func):
        import inspect

        full_sig = inspect.signature(func)
        doc = defopt._parse_docstring(inspect.getdoc(func))
        parameters = []
        for name, param in full_sig.parameters.items():
            if param.name.startswith("_"):
                if param.kind != param.VAR_KEYWORD and param.default is param.empty:
                    raise ValueError(
                        "Parameter {} of {}{} is private but has no default".format(
                            param.name, func.__name__, full_sig
                        )
                    )
            else:
                parameters.append(
                    defopt.Parameter(
                        name=param.name,
                        kind=param.kind,
                        default=param.default,
                        annotation=defopt._get_type(func, param.name),
                        doc=doc.params.get(param.name, defopt._Param(None, None)).text,
                    )
                )
        return full_sig.replace(parameters=parameters)

    defopt.signature = signature

    parser = defopt._create_parser(
        funcs,
        strict_kwonly=False,
        parsers={
            typing.Container[str]: str.split,
            typing.Dict: json.loads,
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
                    "--conf",
                    action=SetConfig,
                    metavar="X",
                    help="Use global configuration (see docs)",
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
                    log.info(f"Writing result to {args.out}")
                    result._dump(args.out, force=True)
                else:
                    log.info(f"Writing result to stdout")
                    result._dump(sys.stdout, force=True)
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
