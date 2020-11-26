import json
import os
import sys
import inspect
import argparse
import typing
import logging as log
from pathlib import Path

import toml


def build(conf, assets=(), base=None, load=None, **kwargs):
    if load:
        assets = parse(load)

    if base is None:
        base = __name__.split(".")[0]

    if assets:
        conf = resolve(conf, assets)
    else:
        assets = {}

    if isinstance(conf, list):
        # don't pass assets here, should already be resolved
        conf = [build(v, base=base, **kwargs) for v in conf]
    if isinstance(conf, dict):
        # don't pass assets here, should already be resolved
        conf = {k: build(v, base=base, **kwargs) for k, v in conf.items()}
        if "class" in conf:
            clsname = conf["class"]
            try:
                mod, name = None, conf.pop("class")
                if "." in name:
                    mod, name = name.rsplit(".", 1)
                if base and mod:
                    mod = f"{base}.{mod}"
                mods = sys.modules[mod or base]
                assert hasattr(mods, name), f"Class '{name}' not found in {mod}!"
                cls = getattr(mods, name)
                cls_params = inspect.signature(cls).parameters
                cls_has_kwargs = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in cls_params.values()
                )

                kwargs = {
                    k: v
                    for k, v in {**conf, **kwargs}.items()
                    if (k in cls_params) or cls_has_kwargs
                }
                obj = cls(**kwargs)
                if "name" in conf:
                    obj.name = conf["name"]
                return obj
            except Exception as e:
                log.error(f"Error when constructing {clsname} from {conf}: {e}")
                raise e
    return conf


def resolve(conf, assets):
    name = None
    if isinstance(conf, str) and conf in assets:
        name = conf
        conf = assets[name]
    elif isinstance(conf, dict) and "resolve" in conf:
        conf = dict(conf)
        name = conf.pop("resolve")
        assert name in assets, f"Asset {name} not found in {tuple(assets)}!"
        if isinstance(assets[name], dict):
            conf = {**assets[name], **conf}
        else:
            conf = assets[name]

    if isinstance(conf, dict):
        if "resolve" in conf:
            conf = resolve(conf, assets)
        conf = {k: (resolve(v, assets) if k != "name" else v) for k, v in conf.items()}
        if name:
            conf = {**conf, "name": name}
    if isinstance(conf, list):
        conf = [resolve(v, assets) for v in conf]

    return conf


def parse(val):
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
    attempts = {}
    for cpi, string_parse in config_parsers.items():
        try:
            conf = string_parse(val)  # type: ignore
            for a in conf.pop("assets", []):
                conf[a.pop("name")] = a
            return conf
            break
        except Exception as err:
            attempts[cpi] = err
    for cpi, e in attempts.items():
        name = val.replace("\n", "")
        name = name[:20] + "..." if len(name) > 20 else name
        log.debug(f"Did not parse {name} with parser {cpi} due to error {e}")
    return val
