import os, sys, glob, csv, json, html, urllib
import trident

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

def read_t2d(root, v=2):
    def get_name(fpath):
        return os.path.basename(fpath).split('.')[0]
    
    # Rows
    table_rows = {}
    for fname in glob.glob(os.path.join(root, 'tables', '*')):
        tablefile = open(fname, 'rb').read().decode('utf-8', errors='ignore')
        if v == 1:
            table_rows[get_name(fname)] = list(csv.reader(tablefile.splitlines()))
        if v == 2:
            table_rows[get_name(fname)] = list(zip(*json.loads(tablefile).get('relation', [])))
    
    # Entities
    table_entities = {}
    for fname in glob.glob(os.path.join(root, 'instance', '*')):
        name = get_name(fname)
        table_entities[name] = {}
        for uri, celltext, rownum in csv.reader(open(fname)):
            uri = html.unescape(urllib.parse.unquote(uri)).replace('/page/', '/resource/')
            if v == 2:
                uri = urllib.parse.unquote(uri)
            uri = fixuri.get(uri, uri)
            table_entities[name][rownum] = uri
    
    # Classes
    table_class = {}
    classfile = os.path.join(root, 'classes_GS.csv' if v==2 else 'classes_instance.csv')
    if os.path.exists(classfile):
        for row in csv.reader(open(classfile)):
            if len(row) == 3:
                fname, label, uri = row
            else:
                fname, label, uri, keys = row
            table_class[get_name(fname)] = uri
    
    # Properties
    table_properties = {}
    table_keycol = {}
    for fname in glob.glob(os.path.join(root, 'property', '*')):
        name = get_name(fname)
        table_properties[name] = {}
        for row in csv.reader(open(fname)):
            if len(row) == 4:
                uri, header, iskey, colnum = row
            else:
                uri, header, colnum = row
            table_properties[name][colnum] = uri
            if iskey.lower() == 'true':
                table_keycol[name] = colnum
            
    
    for name in set(table_rows) | set(table_entities) | set(table_class) | set(table_properties):
        yield {
            'name': name,
            'rows': table_rows.get(name, []),
            'entities': table_entities.get(name, {}),
            'class': table_class.get(name),
            'properties': table_properties.get(name, {}),
            'keycol': table_keycol.get(name),
            'numheaderrows': 1,
        }
        
        

kbs = {
    'dbpedia_t2ksubset': {
        'name': 'dbpedia_t2ksubset',
        'path': ['dbpedia_t2ksubset','db'],
        'classuri': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
        'labeluri': 'http://www.w3.org/2000/01/rdf-schema#label',
    },
    'dbpedia_2014_part2': {
        'name': 'dbpedia_2014_part2',
        'path': ['dbpedia','2014_part2'],
        'classuri': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
        'labeluri': 'http://www.w3.org/2000/01/rdf-schema#label',
    },
    'dbpedia_2016-10': {
        'name': 'dbpedia_2016-10',
        'path': ['dbpedia','2016-10'],
        'classuri': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
        'labeluri': 'http://www.w3.org/2000/01/rdf-schema#label',
    },
}
class Db(trident.Db):
    def __init__(self, kbpath, **kwargs):
        super().__init__(kbpath)
        self.name = kwargs['name']
        self.p_class = self.lookup_id('<%s>'%kwargs['classuri'])
        self.p_label = self.lookup_id('<%s>'%kwargs['labeluri'])
        self.dtype_double = [
            '<http://www.w3.org/2001/XMLSchema#double>',
        ]
    
    def literal_match(self, literal, surface):
        score, dtype = 0, None
        if '^^' in literal:
            literal, dtype = literal.split('^^', 1)
        elif '@' in literal:
            literal, dtype = literal.split('@', 1)
        literal = literal[1:-1].strip()
        surface = surface.strip()
        
        # TODO: distance functions
        if dtype in self.dtype_double:
            surface, literal = surface.replace(',', ''), literal.replace(',', '')
            try:
                s, l = float(surface), float(literal)
                score = 1 - (abs(s - l) / max(s, l))
            except Exception as e:
                score = 0
        elif dtype and dtype[0] == '<':
            # Non-double typed literals should match exactly
            s, l = surface.lower(), literal.lower()
            score = 1 if (s == l) else 0
            
        elif surface and literal:
            # Strings may match approximately
            
            s, l = surface.lower(), literal.lower()
            score = 1 if (s == l) else 0
            if not score:
                score = 0.5 if ((s in l) or (l in s)) else 0
        
        return score, literal, dtype
            
    
    def match(self, e, p, surface):
        for o in self.o(e,p):
            s = self.lookup_str(o)
            if s[0] == '"':
                return self.literal_match(s, surface)
            else:
                matches = []
                for label in self.o(o, self.p_label):
                    label = self.lookup_str(label)
                    matches.append( self.literal_match(label, surface) )
                if matches:
                    return max(matches, key=lambda x:x[0])
        return 0, None, None
