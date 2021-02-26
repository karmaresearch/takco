import os, sys, glob, csv, json, html, urllib
import logging as log

import rdflib
from rdflib import URIRef, Literal

# from .. import link


def config_init_class(self, **context):
    import inspect, copy

    if isinstance(self, dict) and ("class" in self):
        self = copy.deepcopy(self)
        if inspect.isclass(context.get(self["class"])):
            c = context[self.pop("class")]
            kwargs = {k: config_init_class(v, **context) for k, v in self.items()}
            return c(**kwargs)
    return self


def load_kb(config):
    config = {k: config_init_class(v, **link.__dict__) for k, v in config.items()}
    return link.RDFSearcher(**config)
