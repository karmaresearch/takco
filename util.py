def rdf_datatype(node):
    '''From RDF datatype string, get (root, type) pair'''
    if not node:
        return '', ''
    if node[0] == '<':
        if node[-1] == '>':
            return node[1:-1], 'uri' # uri
    elif node[0] == '"':
        if node[-1] == '"':
            return node[1:-1], 'str' # string
        elif node[-1] == '>' and '^^' in node:
            lit, dtype = node.rsplit('^^', 1)
            return lit[1:-1], dtype[1:-1] # typed literal
        elif '@' in node:
            lit, lang = node.rsplit('@', 1)
            return lit[1:-1], '@%s'%lang # language string
        else:
            return node[1], 'str' # string
    return '', ''

import urllib.parse, string
SAFE = ''.join(sorted(set(string.printable) - set('?"\\`%^')))
quote = lambda uri: urllib.parse.quote(uri, SAFE)
unquote = lambda uri: urllib.parse.unquote(uri)


import re
expansion = re.compile('[\(,].*')
def label_variants(s):
    '''Expand a label into variants by removing bracketed parts or parts after a comma'''
    # e.g. "London, UK" or "London (song)"
    yield s
    s, n = expansion.subn('', s)
    if n and s:
        yield s.strip()

def jaccard(a,b):
    a,b = set(a), set(b)
    return len(a&b) / len(a|b) if a|b else 0

import re
token_pattern = re.compile(r"(?u)\b\w+\b")
def tokenize(s):
    return tuple(token_pattern.findall(s.lower()))

def tokenjaccard(a,b):
    a,b = set(tokenize(a)), set(tokenize(b))
    return len(a&b) / len(a|b) if a|b else 0

class BloomDict():
    def __init__(self, size):
        self.size, self.bloomdict = size, {}
    def add(self, key, *items):
        self.bloomdict.setdefault(key, 0)
        for t in items: self.bloomdict[key] += 2**(hash(t) % self.size)
    def has(self, key, t):
        bits = self.bloomdict.get(key, 0)
        if bits: return bool(bits & (2**(hash(t) % self.size)))
        return False

import csv, html
class WebTable():
    _whitespace = re.compile('\s+')
    _tokenize = lambda s: ' '.join(s.split())
    
    num = re.compile('[-0-9.]')
    def _col_isnumeric(col): # column is numeric if >half of the values are numeric
        col = [c for c in col if c.strip()]
        numratio = [(sum(1 for _ in WebTable.num.finditer(c)) / len(c)) if c else 0 for c in col]
        return (((sum(numratio) / len(col)) if col else 0) > 0.5)
    def _col_stdlen(col): # standard deviation of cell lengths
        lens = [len(c.strip()) for c in col if c.strip()]
        if lens:
            meanlen = sum(lens) / len(lens)
            return sum([abs(meanlen-len(c)) for c in col if c]) / len(lens)
        return 0
    def _col_uniqueness(col):
        return len(set([c for c in col if c])) / len(col)
    
    def __init__(self, header, rows, max_uniqueness_threshold=0, uniqueness_ratio_threshold=.8):
        self.header = header
        self.rows = rows
        self.cols = list(zip(*rows))
        self.splitcells()
        
        # Column stats
        self.col_isnumeric = {c:WebTable._col_isnumeric(col) for c,col in enumerate(self.cols)}
        self.col_stdlen = {c:WebTable._col_stdlen(col) for c,col in enumerate(self.cols)}
        self.col_uniqueness = {c:WebTable._col_uniqueness(col) for c,col in enumerate(self.cols)}
        self.max_uniqueness = max((self.col_uniqueness[c]
                              for c,_ in enumerate(self.cols) 
                              if not self.col_isnumeric[c]), 
                             default=0)
        self.keycol = None
        if self.max_uniqueness >= max_uniqueness_threshold:
            for c,col in enumerate(self.cols):
                if ((not self.col_isnumeric[c]) and 
                    (self.col_stdlen[c] > 0) and 
                    ((self.col_uniqueness[c]/self.max_uniqueness) >= uniqueness_ratio_threshold)
                   ):
                    self.keycol = c
                    break
    
    def print_colstats(self):
        print(f'Max uniqueness = {self.max_uniqueness}')
        for c, col in enumerate(self.cols):
            num, stdlen, uniq = self.col_isnumeric[c], self.col_stdlen[c], self.col_uniqueness[c]
            print(f'{c} {self.header[c]:>20s} num:{num:1d} stdlen:{stdlen:2.2f} uniq:{uniq:.2f}')
        
    
    def enumerate_keys(self):
        if self.keycol != None:
            for r,row in enumerate(self.rows_split):
                yield r, next( iter(row[self.keycol]), '' )
    
    def splitcells(self):
        # split cells with mutliple values of the form {foo|bar} into sets
        self.rows_split = [[c for c in row] for row in self.rows]
        for r,row in enumerate(self.rows):
            for c,cells in enumerate(row):
                cells = html.unescape(cells).replace('}','{')
                cells = [v for c in next(csv.reader([cells], delimiter='|', quotechar='{')) for v in c.split('|')]
                self.rows_split[r][c] = cells
    
    def from_csv(fname, header=True, *args, **kwargs):
        # Read CSV, assuming the first row is the header
        if header:
            header, *rows = list(csv.reader(open(fname,'r')))
        else:
            rows = list(csv.reader(open(fname,'r')))
        rows = [[WebTable._whitespace.sub(' ', c).replace('NULL', '') for c in r] for r in rows]
        rows = [row for row in rows if any(row)]
        self = WebTable(header, rows, *args, **kwargs)
        return self

    

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
def get_pr_curve_scores(df, gold_col, pred_col):
    
    y_test, y_score = df[gold_col].map(int), df[pred_col]
    
    if not (list(y_test) or list(y_score)):
        return 0,0,0,0, 0,0,0, [],[]
    ap = average_precision_score(list(y_test), list(y_score))
    
    n_correct = sum((y_score > 0) & (y_test > 0))
    n_pred = sum(y_score > 0)
    n_gold = sum(y_test > 0)
    
    p = n_correct / n_pred if n_pred else 0
    r = n_correct / n_gold if n_gold else 0
    f = 2 / ((1/p) + (1/r)) if p and r else 0
    
    p_curve, r_curve, _ = precision_recall_curve(list(y_test), list(y_score))
    return p,r,f,ap, n_correct,n_pred,n_gold, p_curve, r_curve


def plot_pr_curve():
    pass


import trident
class Db(trident.Db):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 

        
