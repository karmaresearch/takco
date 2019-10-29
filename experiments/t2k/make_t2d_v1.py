import sys, os, csv, urllib.parse, html

# Download t2d-v1 from http://webdatacommons.org/webtables/goldstandard.html#toc5

# Classes
with open('./data/v1/gs_class.csv', 'w') as fw:
    with open('./data/v1/classes_instance.csv') as fr:
        cw = csv.writer(fw)
        for row in csv.reader(fr):
            name = row[0].replace('.tar.gz', '.csv')
            cw.writerow([name, row[1], 'true'])

with open('./data/v1/gs_instance.csv', 'w') as fw:
    cw = csv.writer(fw, quoting=csv.QUOTE_ALL)
    for fname in os.listdir('./data/v1/entities_instance'):
        with open('./data/v1/entities_instance/%s' % fname) as fr:
            for row in csv.reader(fr):
                uri = row[0]
                uri = html.unescape(urllib.parse.unquote(uri)).replace('/page/', '/resource/')
                cw.writerow(['%s~Row%s' % (fname, int(row[2])-1), uri, 'TRUE'])
                
with open('./data/v1/gs_property.csv', 'w') as fw:
    cw = csv.writer(fw)
    for fname in os.listdir('./data/v1/attributes_instance'):
        with open('./data/v1/attributes_instance/%s' % fname) as fr:
            for row in csv.reader(fr):
                cw.writerow(['%s~Col%s' % (fname, row[3]), row[0], 'true'])