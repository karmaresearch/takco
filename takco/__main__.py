"""
tacko is a modular system for extracting knowledge from tables.
"""
__version__ = "0.1.0"

import typing
from pathlib import Path
import logging as log

from . import *
from .util import *

executor = TableSet


def get_executor(name):
    if name:
        config = {}
        if isinstance(name, dict) and ("name" in name):
            config = dict(name)
            name = config.pop("name")

        if name == "dask":
            from dask.distributed import LocalCluster, Client

            config.setdefault("scheduler_port", 8786)
            config.setdefault("dashboard_address", ":8787")
            config.setdefault("n_workers", 4)
            config.setdefault("threads_per_worker", 1)
            # config.setdefault('memory_target_fraction', .9)

            log.info(f"Starting dask cluster with {config}")
            cluster = LocalCluster(**config)
            log.info(f"Started {cluster}")

            client = Client(cluster)

            return DaskHashBag
        elif name == "tqdm":
            return TqdmHashBag
        else:
            return TableSet
    else:
        return TableSet


def load_tables(s):
    global executor
    log.debug(f"Loading tables {s} using executor {executor}")
    return executor._load(s)


def run(
    pipeline: Config,
    workdir: Path = None,
    kbdir: Path = None,
    datadir: Path = None,
    resourcedir: Path = None,
    assets: typing.Dict = None,
    kbs: typing.Dict = None,
    force: bool = False,
    step_force: int = None,
    cache: bool = False,
    executor: str = None,
):
    """Run entire pipeline
    
    Args:
        pipeline: Pipeline config
        workdir: Working directory (also for cache)
        kbdir: Knowledge Base directory
        datadir: Data directory
        resourcedir: Resource directory
        assets: Asset definitions
        kbs: Knowledge Base definitions
        force: Force execution of steps if cache files are already present
        step_force: Force this step number and later
        cache: Cache intermediate results
    """
    pipeline = Config(pipeline)

    executor = executor or pipeline.pop("executor", None)
    executor = (
        (executor or TableSet) if isinstance(executor, type) else get_executor(executor)
    )

    from inspect import signature

    if "name" in pipeline:
        name = pipeline["name"]
    elif "path" in pipeline:
        name = Path(pipeline["path"]).name.split(".")[0]
    else:
        import datetime

        name = "takco-run-" + str(datetime.datetime.now().isoformat())

    if "workdir" in pipeline:
        workdir = pipeline["workdir"]

    if not workdir:
        workdir = Path(pipeline.get("path", ".")).parent.resolve() / Path(name)
    workdir = Path(workdir)
    if cache:
        workdir.mkdir(exist_ok=True, parents=True)

    config = dict(
        datadir=datadir,
        resourcedir=resourcedir,
        workdir=workdir,
        assets=assets or {},
        kbs=kbs or {},
        cache=cache,
        executor=executor,
    )
    for k, v in config.items():
        if k in pipeline:
            config[k] = v or pipeline[k]

    def wrap_step(stepfunc, stepargs, stepdir):
        if config.get("cache"):
            import json, shutil

            shutil.rmtree(stepdir, ignore_errors=True)
            stepdir.mkdir(exist_ok=True, parents=True)
            tablefile = Path(stepdir) / Path("*")

            log.info(f"Writing cache to {tablefile}")
            return stepfunc(**stepargs)._dump(tablefile)
        else:
            return stepfunc(**stepargs)

    log.info(f"Running pipeline '{name}' in {workdir}")
    tables = []
    for si, stepargs in enumerate(pipeline.get("step", [])):
        if "step" in stepargs:
            steptype = stepargs.pop("step")
            stepname = f"{si}-{stepargs.get('name', steptype)}"
            stepdir = Path(workdir) / Path(stepname)

            nodir = (not stepdir.exists()) or (not any(stepdir.iterdir()))
            if force or (si >= step_force) or nodir:

                stepfunc = getattr(TableSet, steptype)
                if stepfunc:
                    sig = signature(stepfunc)
                    local_config = dict(tables=tables, **config)
                    for k, v in local_config.items():
                        if (k in sig.parameters) and (k not in stepargs):
                            stepargs[k] = v
                    log.info(f"Chaining pipeline step {stepname}: {stepargs}")
                    tables = wrap_step(stepfunc, stepargs, stepdir)
                else:
                    log.warning(f"Pipeline step type '{steptype}' does not exist")
            else:
                log.warn(f"Skipping step {stepname}")
                tables = executor._load(str(stepdir) + "/*")
        else:
            log.warn(f"Pipeline step {si} has no step type!")
    return tables


def main():
    import os, sys, defopt, json, toml, types, logging, argparse

    class SetConfig(argparse.Action):
        def __init__(self, option_strings, dest, nargs="?", **kwargs):
            super(SetConfig, self).__init__(option_strings, dest, nargs="?", **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            config = {}
            if self.dest and values:
                conf = Config(values)
                if conf:
                    config.update(conf)
                    logging.info(f"Loaded config {conf}")
            elif Path("config.toml").exists():
                config.update(toml.load(Path("config.toml").open()))
                log.info(f"Loaded local config.toml")
            else:
                config.update(os.environ)
                log.info(f"Loaded config from environment")

            if "executor" in config:
                global executor
                executor = get_executor(config.pop("executor"))
                config["executor"] = executor
                log.debug(f"Set config to use executor {executor}")

            for k, v in config.items():
                setattr(namespace, k, v)

    class SetVerbosity(argparse.Action):
        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            if nargs is not None:
                raise ValueError("nargs not allowed")
            super(SetVerbosity, self).__init__(option_strings, dest, nargs=0, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            if self.dest:
                loglevel = getattr(logging, self.dest.upper(), logging.WARNING)
                logging.getLogger().setLevel(loglevel)
            else:
                ll = os.environ.get("LOGLEVEL", "").upper()
                loglevel = getattr(logging, ll, logging.WARNING)
                logging.getLogger().setLevel(loglevel)
                logfile = os.environ.get("LOGFILE", None)
                logging.getLogger().addHandler(logging.FileHandler(logfile))
            log.info(f"Set log level to {logging.getLogger().getEffectiveLevel()}")

    funcs = (
        run,
        appcache,
        TableSet.dataset,
        TableSet.extract,
        TableSet.reshape,
        TableSet.cluster,
        TableSet.integrate,
        TableSet.link,
        TableSet.score,
        TableSet.triples,
    )
    parser = defopt._create_parser(
        funcs,
        strict_kwonly=False,
        parsers={
            typing.Container[str]: str.split,
            typing.Dict: json.loads,
            Config: Config,
            TableSet: load_tables,
            HashBag: HashBag._load,
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
