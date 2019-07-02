import sys, os, csv, urllib.parse, html

# Download t2d-v1ex from http://webdatacommons.org/webtables/goldstandard.html#toc5

# Classes
with open('./data/v1ex/classes_GS.csv') as fr:
    with open('./data/v1ex/gs_class.csv', 'w') as fw:
        cw = csv.writer(fw)
        for row in csv.reader(fr):
            name = row[0].replace('.tar.gz', '.csv')
            cw.writerow([name, row[1], 'true'])

with open('./data/v1ex/gs_instance.csv', 'w') as fw:
    cw = csv.writer(fw, quoting=csv.QUOTE_ALL)
    for fname in os.listdir('./data/v1ex/instance'):
        with open('./data/v1ex/instance/%s' % fname) as fr:
            for row in csv.reader(fr):
                uri = row[0]
                uri = html.unescape(urllib.parse.unquote(uri)).replace('/page/', '/resource/')
                cw.writerow(['%s~Row%s' % (fname, int(row[2])-1), uri, 'TRUE'])
                
with open('./data/v1ex/gs_property.csv', 'w') as fw:
    cw = csv.writer(fw)
    for fname in os.listdir('./data/v1ex/property'):
        with open('./data/v1ex/property/%s' % fname) as fr:
            for row in csv.reader(fr):
                try:
                    cw.writerow(['%s~Col%s' % (fname, row[-1]), row[0], 'true'])
                except:
                    print(fname, row)

import json
for fname in os.listdir('./data/v1ex/tables'):
    if not fname.endswith('json'):
        continue
    try:
        data = json.loads(open('./data/v1ex/tables/%s' % fname, 'rb').read().decode('utf-8', errors='ignore'))
        relation = data.get('relation', None)
        if relation:
            with open('./data/v1ex/tables/%s' % fname.replace('.json', '.csv'), 'w', encoding='utf-8') as fw:
                cw = csv.writer(fw, quoting=csv.QUOTE_ALL)
                rows = zip(*relation)
                for row in rows:
                    cw.writerow(row)
    except UnicodeDecodeError:
        print(fname)
        
    