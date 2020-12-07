import typing

class LinkedString(str):
    links: typing.Tuple[typing.Tuple[int, int, str], ...] = ()
    
    def __new__(cls, value, links=None):
        if isinstance(value, LinkedString):
            value, links = value.text, value.links
        self = super().__new__(cls, value)
        if links is not None:
            strlen = len(self)
            links = tuple(
                (max(0, int(start)), min(int(end), strlen), val)
                for start, end, val in links
                if start < end
            )
        else:
            links = ()
        self.links = links
        return self
    
    @property
    def text(self):
        return str(self)
    
    def __repr__(self):
        return f"LinkedString('{self}', links={self.links.__repr__()})"
    
    def to_html(self):
        text = str(self)
        val = ''
        last = None
        for s, e, v in self.links:
            val += text[last:s] + f'<a href="{v}">{text[s:e]}</a>'
            last = e
        val += text[last:]
        return val
    
    def _repr_html_(self):
        return self.to_html()
    
    def __hash__(self):
        return hash(str(self)) + hash(self.links)
    
    def __eq__(self, string):
        if isinstance(string, LinkedString):
            return str(self) == str(string) and self.links == string.links
        return str(self) == string
    
    def __getitem__(self, index):
        if not isinstance(index, slice):
            index = slice(index, index+1)
        l = len(self)
        s, e = index.start, index.stop
        s = (s if s > 0 else max(0, l+s)) if s is not None else 0
        e = (e if e > 0 else max(0, l+e)) if e is not None else l
        links = tuple(
            (ls-s, le-s, lv)
            for ls, le, lv in self.links
            if s < le and e > ls
        )
        return self.__class__(str(self)[index], links=links)
    
    def __add__(self, other):
        if isinstance(other, LinkedString):
            l = len(self)
            shiftlinks = tuple((s+l, e+l, v) for s, e, v in other.links)
            return self.__class__(str(self) + str(other), links=self.links + shiftlinks)
        elif isinstance(other, str):
            return self.__class__(str(self) + other, links=self.links)
        return self.__class__(str(self) + str(other), links=self.links)
    
    def __radd__(self, other):
        l = len(other)
        shiftlinks = tuple((s+l, e+l, v) for s, e, v in self.links)
        if isinstance(other, LinkedString):
            return self.__class__(str(other) + str(self), links=other.links + shiftlinks)
        elif isinstance(other, str):
            return self.__class__(other + str(self), links=shiftlinks)
        return self.__class__(str(other) + str(self), links=shiftlinks)

    def join(self, seq):
        iter_seq = iter(seq)
        out = LinkedString(next(iter_seq, ''))
        for i in iter_seq:
            out = LinkedString(out) + self + LinkedString(i)
        return out