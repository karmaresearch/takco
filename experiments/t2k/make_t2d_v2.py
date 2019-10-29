import sys, os, csv, urllib.parse, html

# Download t2d-v2 from http://webdatacommons.org/webtables/goldstandard.html#toc5

fixuri = {
    'http://dbpedia.org/resource/Andr�-Marie_Amp�re':'http://dbpedia.org/resource/André-Marie_Ampère',
    'http://dbpedia.org/resource/Erwin_Schr�dinger': 'http://dbpedia.org/resource/Erwin_Schrödinger',
    'http://dbpedia.org/resource/Leopold_Ru�icka': 'http://dbpedia.org/resource/Leopold_Ružička',
    'http://dbpedia.org/resource/Mu?ammad_ibn_Musa_al-Khwarizmi': 'http://dbpedia.org/resource/Muhammad_ibn_Musa_al-Khwarizmi',
    'http://dbpedia.org/resource/Ren�_Descartes': 'http://dbpedia.org/resource/René_Descartes',
    'http://dbpedia.org/resource/Richard_Willst�tter': 'http://dbpedia.org/resource/Richard_Willstätter',
    'http://dbpedia.org/resource/Teresa_of_�vila': 'http://dbpedia.org/resource/Teresa_of_Ávila',
    'http://dbpedia.org/resource/Th�r�se_of_Lisieux': 'http://dbpedia.org/resource/Thérèse_of_Lisieux',
    'http://dbpedia.org/resource/Vincent_of_L�rins': 'http://dbpedia.org/resource/Vincent_of_Lérins',
    'http://dbpedia.org/resource/Wilhelm_R�ntgen': 'http://dbpedia.org/resource/Wilhelm_Röntgen',
    'http://dbpedia.org/resource/�thelberht_of_Kent': 'http://dbpedia.org/resource/Æthelberht_of_Kent',
    'http://dbpedia.org/resource/�tienne_Lenoir': 'http://dbpedia.org/resource/Étienne_Lenoir'
}

# Classes
with open('./data/v2//gs_classes.csv') as fr:
    with open('./data/v2//gs_class.csv', 'w') as fw:
        cw = csv.writer(fw)
        for row in csv.reader(fr):
            name = row[0].replace('.tar.gz', '.csv')
            cw.writerow([name, row[1], 'true'])

with open('./data/v2//gs_instance.csv', 'w') as fw:
    cw = csv.writer(fw, quoting=csv.QUOTE_ALL)
    for fname in os.listdir('./data/v2//instance'):
        with open('./data/v2//instance/%s' % fname) as fr:
            for row in csv.reader(fr):
                uri = row[0]
                uri = html.unescape(urllib.parse.unquote(uri)).replace('/page/', '/resource/')
                uri = urllib.parse.unquote(uri)
                uri = fixuri.get(uri, uri)
                cw.writerow(['%s~Row%s' % (fname, int(row[2])-1), uri, 'TRUE'])
                
with open('./data/v2//gs_property.csv', 'w') as fw:
    cw = csv.writer(fw)
    for fname in os.listdir('./data/v2//property'):
        with open('./data/v2//property/%s' % fname) as fr:
            for row in csv.reader(fr):
                try:
                    cw.writerow(['%s~Col%s' % (fname, row[-1]), row[0], 'true'])
                except:
                    print(fname, row)

import json
for fname in os.listdir('./data/v2//tables'):
    if not fname.endswith('json'):
        continue
    try:
        data = json.loads(open('./data/v2//tables/%s' % fname, 'rb').read().decode('utf-8', errors='ignore'))
        relation = data.get('relation', None)
        if relation:
            with open('./data/v2//tables/%s' % fname.replace('.json', '.csv'), 'w', encoding='utf-8') as fw:
                cw = csv.writer(fw, quoting=csv.QUOTE_ALL)
                rows = zip(*relation)
                for row in rows:
                    cw.writerow(row)
    except UnicodeDecodeError:
        print(fname)
        
    
