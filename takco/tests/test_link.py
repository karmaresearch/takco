import pytest
import typing
import logging as log

import takco

@pytest.fixture(scope="session")
def table():
    return {
        'tableData': [
            [
                {'text': 'Amsterdam'}
            ]
        ]
    }
    
def test_load(table: typing.Dict):
    assert bool(table), f"Could not load any table"


def test_entity(table: typing.Dict):
        
    linker = takco.link.First(takco.link.DBpediaLookup())
    for t in takco.link.link([table], linker=linker):
        assert t
        log.info(t)
    
    

if __name__ == "__main__":
    import defopt, json
    
    log.getLogger().setLevel(log.DEBUG)
    defopt.run(
        [test_entity],
        parsers={typing.Dict: json.loads, typing.Any: json.loads},
        strict_kwonly=False,
    )
