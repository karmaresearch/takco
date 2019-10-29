import sys, requests, glob, urllib.parse

p_uri = 'http://www.w3.org/2002/07/owl#sameAs'

if __name__ == '__main__':
    import sys, os.path, json
    try:
        _, candidate_dir, done = sys.argv
    except Exception as e:
        print('Usage: lookup_dbpedia_wikidata_uri.py [candidate_dir] [done]')
        raise e
        
    import pandas as pd
    fglob = os.path.join(candidate_dir, '*.csv')
    candidates = pd.concat([pd.read_csv(fname) for fname in glob.glob(fglob)])
    uris = set(candidates['entity'])
    print('%d uris total' % len(uris), file=sys.stderr)

    wikidata = set([line.split()[0][1:-1] for line in open(done)])
    print('%d not found yet' % len(uris - wikidata), file=sys.stderr)
    
    uris = uris - wikidata
    print('Looking up %d URIs' % len(uris), file=sys.stderr)

    for uri in uris:
        data_uri  = uri.replace('http://dbpedia.org/resource/', 'http://dbpedia.org/data/') + '.ntriples'

        res = requests.get(data_uri)
        triples = [line for line in res.text.splitlines() if 'owl#sameAs' in line and 'http://www.wikidata.org/entity/' in line]
        for triple in triples:
            print(triple)
        if not triples:
            redir = [line for line in res.text.splitlines() if 'wikiPageRedirects' in line]
            if redir:
                redir = redir[0].split()[2][1:-1].encode().decode('unicode-escape')
                print('Redirected: %s to %s' % (uri, redir), file=sys.stderr)
                redir = redir.replace('http://dbpedia.org/resource/', 'http://dbpedia.org/data/') + '.ntriples'
                res = requests.get(redir)
                triples = [line for line in res.text.splitlines() if 'owl#sameAs' in line and 'http://www.wikidata.org/entity/' in line]
                for triple in triples:
                    print(triple)
        if not triples:
            print('Not found: %s (%d lines)' % (uri, len(res.text.splitlines())), file=sys.stderr)
#             print(res.text, file=sys.stderr)