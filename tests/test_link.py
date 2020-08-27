import pytest
import typing
from pathlib import Path
import logging as log
import asyncio

import toml

from takco.evaluate import dataset
from takco.link import external
from takco.link import baselines

@pytest.fixture(scope="session")
def resources_path():
    return Path("resources")

@pytest.fixture(scope="session")
def config():
    return toml.load(Path("config.toml").open())

@pytest.fixture(scope="session")
def datadir_path(tmpdir_factory: Path):
    return tmpdir_factory.mktemp("data")

@pytest.fixture(scope="session")
def table(resources_path: Path, datadir_path: Path, config: typing.Dict):
    for name, params in config.get('datasets', {}).items():
        dsdir = Path(datadir_path) / Path(name)
        dsdir.mkdir(exist_ok=True, parents=True)
        d = dataset.load(dsdir, resourcedir=resources_path, **params )
        for table in d.tables:
            return table
    
def test_load(table):
    assert bool(table), f"Could not load any table"


def test_entity(table: typing.Dict):
        
    model = baselines.First(external.DBpediaLookup())
    
    searches = [
        (ri, ci, cell)
        for ri,row in enumerate(table['rows'])
        for ci,cell in enumerate(row)
    ]
    futures = [dl.search_entities(cell, limit=1) for _,_,cell in searches]
    results = asyncio.get_event_loop().run_until_complete(asyncio.gather(*futures))
            
    entities = {}
    for (ri,ci, cell), ents in zip(searches, results):
        if ents:
            ent = next(iter(ents))
            if 'uri' in ent:
                entities.setdefault(str(ci), {})[str(ri)] = ent['uri']
    
    log.info(entities)

if __name__ == "__main__":
    import defopt, json
    
    log.getLogger().setLevel(log.DEBUG)
    defopt.run(
        [v for k, v in locals().items() if callable(v) and v.__module__ == __name__],
        parsers={typing.Dict: json.loads, typing.Any: json.loads},
        strict_kwonly=False,
    )
