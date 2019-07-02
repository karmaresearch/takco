import sys, os
try:
    _, host, typ, index = sys.argv
except Exception as e:
    print('Usage: python es_index.py [host] [type] [index]')
    raise e
    
action = {
    '_type': typ,
    '_index': index
}
    
from elasticsearch import Elasticsearch, helpers
es = Elasticsearch(timeout=10000)

es.indices.delete(index=index, ignore=[400, 404])
body = {
    "settings": {
        "index" : {
            "similarity": {
              "scripted_idf": {
                "type": "scripted",
                "script": {
                  "source": "double idf = Math.log((field.docCount+1.0)/(term.docFreq+1.0)) + 1.0; return query.boost * idf;"
                }
              }
            }
        }
    },
    "mappings": {
        typ:{
            "properties": {
                "ids": { "type": "keyword"  },
                "p": { "type": "keyword"  },
                "label": { "type":"text", "similarity" : "scripted_idf" },
                
            }
        }
    }
}
es.indices.create(index=index, body=body)



import sys, json
def stream():
    for line in sys.stdin:
        try:
            doc = dict(action)
            doc.update(json.loads(line))
            yield doc
        except Exception as e:
            print(line, file=sys.stderr)
            print(e, file=sys.stderr)

results = helpers.parallel_bulk(es, stream(), thread_count=8)
for i, (status, r) in enumerate(results):
    if not status:
        print(r, file=sys.stderr)
    if 0 == i % 1000:
        print('indexed %10d' % i, end='\r', file=sys.stderr)
        sys.stderr.flush()

# import time
# time.sleep(1)

# res = es.search(index=index, body={"query": {"match_all": {}}})
# print("Got %d Hits" % res['hits']['total'], file=sys.stderr)

# res = es.search(index=index, body={
#   "query": {
#     "multi_match" : {
#       "query":      "usa",
#       "type":       "cross_fields",
#       "operator":   "or",
#     }
#   }
# })
# print("Got %d usa Hits without" % res['hits']['total'], file=sys.stderr)

# res = es.search(index=index, body={
#   "query": {
#     "multi_match" : {
#       "query":      "usa",
#       "type":       "cross_fields",
#       "operator":   "or",
#       "analyzer": "myanalyzer"
#     }
#   }
# })
# print("Got %d usa Hits with" % res['hits']['total'], file=sys.stderr)