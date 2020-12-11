import logging as log

import takco

def run():
    pass

if __name__ == "__main__":
    import defopt, json
    
    log.getLogger().setLevel(log.DEBUG)
    defopt.run(
        [run],
        parsers={typing.Dict: json.loads, typing.Any: json.loads},
        strict_kwonly=False,
    )
