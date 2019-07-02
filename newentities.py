import pandas as pd
TYPE_URI = 'type'

def parse_cell(cell):
    cell = str(cell)
    if cell[0] == '{' and cell[-1] == '}':
        for part in cell[1:-1].split('|'):
            yield part
    elif cell == '&nbsp;':
        return
    else:
        yield cell

def newentity_triples(fname, instances, schemas, classes):
    name = os.path.basename(fname)
    table = pd.read_csv(fname)
    
    d = pd.DataFrame(instances.loc[name])
    d.set_index('row', inplace=True)
    
    schema = {t.col:t._1 for t in schemas.loc[[name]].itertuples()}
    types = set(c for cs in classes.loc[[name]].itertuples() for c in cs.name.split(','))
    
    for tup in table.itertuples():
        i = tup.Index
        instance = d.loc[i][1] if i in d.index else None
        if not instance:
            row_id = '%s~Row%s' % (name, i)
            for t in types:
                yield (row_id, TYPE_URI, t)
            for i, cell in enumerate(list(tup)[1:]):
                if i in schema:
                    for c in parse_cell(cell):
                        yield (row_id, schema[i], c)



if __name__ == '__main__':
    import sys, glob, os
    try:
        _, dir_tables, file_classes, file_instances, file_schemas = sys.argv
    except:
        print('Usage: newentities.py dir_tables file_classes file_instances file_schemas')
        sys.exit(0)
    
    classes = pd.read_csv(file_classes, header=None, index_col = 0)
    if list(classes.index)[0] == 0:
        classes.set_index(1, inplace=True)
    else:
        del classes[2]
    classes.columns = ['name']
    
    schemas = pd.read_csv(file_schemas, header=None)
    schemas['table'] = schemas[0].map(lambda s: s.split('~')[0])
    schemas['col'] = schemas[0].map(lambda s: int(s.split('~Col')[1]))
    del schemas[0]
    schemas.set_index('table', inplace=True)
    
    instances = pd.read_csv(file_instances, header=None)
    instances['table'] = instances[0].map(lambda s: s.split('~')[0])
    instances['row'] = instances[0].map(lambda s: int(s.split('~Row')[1]))
    del instances[0]
    instances.set_index('table', inplace=True)
    
    for fname in glob.glob('%s/*.*' % dir_tables):
        for s,p,o in newentity_triples(fname, instances, schemas, classes):
            print(s,p,o, sep='\t')