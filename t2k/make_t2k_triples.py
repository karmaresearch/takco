import sys, os
from subprocess import call

try:
    _, gold_dir, results_dir = sys.argv
    gold_dir = os.path.realpath(gold_dir)
    results_dir = os.path.realpath(results_dir)
except Exception as e:
    print('Usage: make_t2k_triples.py [gold_dir] [results_dir]')
    raise(e)

triple_dir = os.path.join(results_dir, 'triples')



os.chdir(os.path.dirname(os.path.realpath(__file__)))
JAR = "./data/t2kmatch-2.1-jar-with-dependencies.jar"
JAR = os.path.realpath(JAR)
CLS = "de.uni_mannheim.informatik.dws.t2k.match.T2KMatch"

RD = "./data/db/redirects.txt" # ???

import pandas as pd
j = pd.read_csv(os.path.join(results_dir, 'joined_instance.csv'))
j['row'] = (j['table'] + '.csv~Row' + j['rownr'].map(str))
j['gold'] = j['gold'].map(int)

joined_property = pd.read_csv(os.path.join(results_dir, 'joined_property.csv'))
joined_property['col'] = (joined_property['table'] + '.csv~Col' + joined_property['colnr'].map(str))
joined_property.set_index(['col', 'uri'], inplace=True)
joined_property['gold'] = joined_property['gold'].map(int)

for method in j.columns:
    if method not in ['table', 'rownr', 'uri', 'row']:
        method_dir = os.path.join(triple_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        
        property_fname = os.path.realpath(os.path.join(method_dir, 'property.csv'))
        
        m = (method if method in joined_property.columns else 'our')
        
        
        grouped = joined_property.groupby(['table','colnr'])
        hasgold = grouped['gold'].transform('sum').map(bool)
        best_per_col = joined_property[hasgold & (grouped[m].transform('max') == joined_property[m])]
        prop = best_per_col[m].to_frame().reset_index()
        prop[m] = 'true'
        prop[['col', 'uri', m]].to_csv(property_fname, header=False, index=False)
        
        sel = ((j[method] > 0) & (j.groupby('row')[method].transform('max') == j[method]))
        instance_fname = os.path.realpath(os.path.join(method_dir, 'instance.csv'))
        print('%s: %d' % (method, sum(sel)))
        j[sel][['row', 'uri', method]].to_csv(instance_fname, header=False, index=False)
        
        log_file = os.path.join(method_dir, 'log.txt')
        
        args = dict(
            sf = "./data/db/surfaceforms.txt",
            kb = "./data/db/dbpedia/",
            ontology = "./data/db/OntologyDBpedia",
            index = "./data/db/index/",

            web = "./data/v1/tables_instance", # os.path.join(gold_dir, "tables"),
            classGS = os.path.join(gold_dir, "gs_class.csv"),
            identityGS = os.path.join(gold_dir, "gs_instance.csv"),

            schemaGS = property_fname,
            results = method_dir,

            verbose = '',
            cellCandidates = '',

            onlyCheckGold = instance_fname,
        )
        args = [a for k,v in args.items() for a in ('-%s'%k, v or '')]

        with open(log_file, 'w') as fw:
            call(["java", "-Xmx128G", "-cp", JAR, CLS, *args], stdout=fw)
        
        ntriples = len(list(open(os.path.join(method_dir, 'extracted_triples-GOLD.csv'))))
        print('Got %d triples' % ntriples)
        

