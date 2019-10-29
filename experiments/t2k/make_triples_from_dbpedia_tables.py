import pandas as pd
import os

TYPE_URI = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'

def yield_triples(dbpedia_table_fname):
    s_type, _ = os.path.splitext(os.path.basename(fname))
    s_type = 'http://dbpedia.org/ontology/' + s_type
    t = pd.read_csv(fname, index_col=0, header=[0,1,2,3], dtype='str')
    for s,row in t.iterrows():
        yield ('<%s>' % s, '<%s>' % TYPE_URI, '<%s>' % s_type)
        for (pred_label, pred, typ_label, typ), vals in row.iteritems():
            if (not pd.isna(vals)) and (not pred_label.endswith('_label')) and vals:
                if vals[0] == '{' and vals[-1] == '}':
                    vals = vals[1:-1].split('|')
                else:
                    vals = [vals]
                for val in vals:
                    if typ_label in ['XMLSchema#string', 'rdf-schema#Literal']:
                        o = '"%s"' % val
                    elif 'XMLSchema' in typ_label:
                        o = '"%s"^^<%s>' % (val, typ)
                    elif val.startswith('http'):
                        o = '<%s>' % val
                    else:
                        o = '"%s"' % val

                    yield ('<%s>' % s, '<%s>' % pred, o)

if __name__ == '__main__':
    import sys
    try:
        _, fname = sys.argv
    except:
        print('Usage: make_triples_from_dbpedia_tables.py <fname.csv>', file=sys.stderr)
        sys.exit(0)
    try:
        print(fname, file=sys.stderr)
        for (s,p,o) in yield_triples(fname):
            print(s,p,o,'.')
    except Exception as e:
        print(fname, file=sys.stderr)
        raise e