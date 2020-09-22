import rdflib
import logging as log

from rdflib.plugins.parsers.ntriples import NTriplesParser


class TridentNode(rdflib.term.Node):
    def __init__(self, id, db):
        self.id = id
        self.db = db

    def __eq__(self, other):
        if isinstance(other, TridentNode):
            return (self.id == other.id) and (self.db == other.db)
        return False

    def __hash__(self):
        return hash((self.id, self.db))

    def __repr__(self):
        return str(self.id)

    @staticmethod
    def _parse(s):
        n = NTriplesParser()
        n.line = s
        return n.object()

    def resolve(self, baseuri=None, ns=None):
        try:
            s = self.db.lookup_str(self.id)
            if ns and s and (":" in s) and (s[1:].split(":", 1)[0] in ns):
                prefix = s[1:].split(":", 1)[0]
                s = s.replace(prefix + ":", ns[prefix])
            elif baseuri and (s[0] == "<") and (s[-1] == ">"):
                s = f"<{baseuri}{s[1:-1]}>"
            return self.__class__._parse(s)
        except:
            return None

    @classmethod
    def from_str(cls, s, db):
        id = db.lookup_id(s)
        if id is not None:
            return cls(id, db)
        else:
            return None


class Trident(rdflib.store.Store):
    def __init__(self, ent_baseuri=None, prop_baseuri=None, ns=None, *args, **kwargs):
        self.db = None
        self.ent_baseuri = ent_baseuri
        self.prop_baseuri = prop_baseuri
        self.ns = ns or {}
        super().__init__(*args, **kwargs)

    def node(self, i, baseuri=None, ns=None):
        return TridentNode(i, self.db).resolve(baseuri=baseuri, ns=ns)

    def id(self, n, baseuri=None, ns=None):
        if isinstance(n, TridentNode):
            return n.id
        else:
            n3 = n.n3().replace("\\\\", "\\")
            if baseuri:
                n3 = n3.replace(baseuri, "")
            for p, ns in self.ns.items():
                n3 = n3.replace(ns, p + ":")
            i = self.db.lookup_id(n3)
            if i is not None:
                return i
            else:
                return self.db.lookup_id(n3 + " ")

    def open(self, configuration: str):
        import trident

        self.db = trident.Db(configuration)
        log.debug(f"Using Trident DB with {len(self)} triples")
        return rdflib.store.VALID_STORE

    def count(self, triple):
        s, p, o = triple
        vs = (s is None) or isinstance(s, rdflib.Variable)
        vp = (p is None) or isinstance(p, rdflib.Variable)
        vo = (o is None) or isinstance(s, rdflib.Variable)
        si = self.id(s, baseuri=self.ent_baseuri, ns=self.ns) if not vs else None
        pi = self.id(p, baseuri=self.prop_baseuri, ns=self.ns) if not vp else None
        oi = self.id(o, baseuri=self.ent_baseuri, ns=self.ns) if not vo else None

        if si and (not pi) and (not oi):
            return self.db.count_s(si)
        elif (not si) and pi and (not oi):
            return self.db.count_p(pi)
        elif (not si) and (not pi) and oi:
            return self.db.count_o(oi)

        elif si and pi and (not oi):
            return self.db.n_o(si, pi)
        elif (not si) and pi and oi:
            return self.db.n_s(pi, oi)

        else:
            ts = self.triples((s, p, o))
            return len(ts) if hasattr(ts, "__len__") else sum(1 for _ in ts)

    def triples(self, triple_pattern, context=None, resolve=True):
        s, p, o = triple_pattern
        vs = (s is None) or isinstance(s, rdflib.Variable)
        vp = (p is None) or isinstance(p, rdflib.Variable)
        vo = (o is None) or isinstance(s, rdflib.Variable)
        si = self.id(s, baseuri=self.ent_baseuri, ns=self.ns) if not vs else None
        pi = self.id(p, baseuri=self.prop_baseuri, ns=self.ns) if not vp else None
        oi = self.id(o, baseuri=self.ent_baseuri, ns=self.ns) if not vo else None

        db = self.db
        funcmap = {
            (vs, vp, vo): lambda: db.all(),
            (not vs, vp, vo): lambda: ((si, rp, ro) for rp, ro in db.po(si)),
            (vs, not vp, vo): lambda: ((rs, pi, ro) for rs, ro in db.os(pi)),
            (vs, vp, not vo): lambda: ((rs, rp, oi) for rp, rs in db.ps(oi)),
            (vs, not vp, not vo): lambda: ((rs, pi, oi) for rs in db.s(pi, oi)),
            (not vs, vp, not vo): lambda: (
                (si, rp, oi) for rp, ro in db.po(si) if oi == ro
            ),
            (not vs, not vp, vo): lambda: ((si, pi, ro) for ro in db.o(si, pi)),
            (not vs, not vp, not vo): lambda: (
                x for x in [(si, pi, oi)] if db.exists(*x)
            ),
        }

        all_tridentnodes = all(isinstance(n, TridentNode) for n in (s, p, o))
        if any(all(cond) for cond in funcmap):
            for cond, func in funcmap.items():
                if all(cond):
                    try:
                        for rs, rp, ro in func():
                            if all_tridentnodes or (not resolve):
                                r = (
                                    TridentNode(rs, self.db),
                                    TridentNode(rp, self.db),
                                    TridentNode(ro, self.db),
                                )
                            else:
                                r = (
                                    self.node(rs, baseuri=self.ent_baseuri, ns=self.ns),
                                    self.node(
                                        rp, baseuri=self.prop_baseuri, ns=self.ns
                                    ),
                                    self.node(ro, baseuri=self.ent_baseuri, ns=self.ns),
                                )
                            yield r, None
                    except TypeError:
                        pass
        else:
            raise rdflib.exceptions.Error(
                f"Could not get triples of {self}" + f"for {triple_pattern}"
            )

    def __len__(self, context=None):
        return self.db.n_triples()
