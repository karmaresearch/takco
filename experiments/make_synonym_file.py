import sys, urllib.parse, html, re

import unicodedata
from collections import defaultdict
unicode_category = defaultdict(set)
for c in map(chr, range(sys.maxunicode + 1)):
    unicode_category[unicodedata.category(c)].add(c)

W = re.compile(r'\w')
for line in sys.stdin:
    line = line.strip().replace(',',' ')
    line = urllib.parse.unquote(line.encode().decode('unicode_escape'))
    print(','.join((t for t in line.split('\t') if W.search(t) and t[0] not in unicode_category['No'])))
    sys.stdout.flush()