import sys, os
from subprocess import call

try:
    _, gold_dir, moreargs = sys.argv
    gold_dir = os.path.realpath(gold_dir)
    moreargs = moreargs.split()
except Exception as e:
    print('Usage: run_t2k_match.py [gold_dir]')
    raise(e)

os.chdir(os.path.dirname(os.path.realpath(__file__)))

results_dir = os.path.realpath('output-' + os.path.basename(gold_dir) + ''.join(moreargs))
os.makedirs(results_dir, exist_ok=True)
log_file = os.path.join(results_dir, 'log.txt')

JAR="./t2kmatch-2.1-jar-with-dependencies.jar"
CLS="de.uni_mannheim.informatik.dws.t2k.match.T2KMatch"

RD="./db/redirects.txt" # ???

args = dict(
    sf = "./db/surfaceforms.txt",
    kb = "./db/dbpedia/",
    ontology = "./db/OntologyDBpedia",
    index = "./db/index/",
    
    web = os.path.join(gold_dir, "tables"),
#     classGS = os.path.join(gold_dir, "gs_class.csv"),
    schemaGS = os.path.join(gold_dir, "gs_property.csv"),
    identityGS = os.path.join(gold_dir, "gs_instance.csv"),

    results = results_dir,
    
    verbose = '',
    cellCandidates = '',
)
args = [a for k,v in args.items() for a in ('-%s'%k, v or '')]
print(' '.join(["java", "-Xmx128G", "-cp", JAR, CLS, *args, *moreargs]))

with open(log_file, 'w') as fw:
#     call(["java", "-Xmx128G", "-cp", JAR, CLS, *args], stdout=fw)
    call(["java", "-Xmx128G", "-cp", JAR, CLS, *args, *moreargs], stdout=fw)


