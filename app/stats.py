
def get_kbgold(kb, table):
    kblinks = get_table_kblinks(kb, table)
    kblinks['novelty'] = get_novelty(kb, kblinks)
    kbinfo = get_kbinfo(kb, kblinks)
    return kblinks, kbinfo

def get_kbinfo(kb, kblinks):
    kbinfo = {}
    props = [p for p in kblinks['properties_id'].values() if p is not None]
    for e in set(props) | set([kblinks['class_id']]):
        if e:
            kbinfo[str(e)] = { 'uri': kb.lookup_str(e), 'size': kb.count_p(e) + kb.count_o(e), }

    for e in set(kblinks['entities_id'].values()):
        if e:
            kbinfo[str(e)] = {
                'uri': kb.lookup_str(e),
                'props': [
                    {'uri':kb.lookup_str(p), 'vals':sorted([kb.lookup_str(l) for l in kb.o(e, p) if e and l])}
                    for p in sorted(set([kb.p_label, kb.p_class]) | set(props))
                ],
                'inlinks': kb.count_o(e),
                'outlinks': kb.count_s(e),
            }
    return kbinfo

def get_table_kblinks(kb, table):
    kblinks = {}
    kblinks['class_id'] = kb.lookup_id('<%s>' % table['class'])
    kblinks['entities_id'] = {str(ri):kb.lookup_id('<%s>'%uri) for ri,uri in table['entities'].items()}
    kblinks['properties_id'] = {str(ci):kb.lookup_id('<%s>'%uri) for ci,uri in table['properties'].items()}

    kblinks['entity_hasclass'] = {
        str(s): kb.exists(s, kb.p_class, kblinks['class_id'])
        for s in kblinks['entities_id'].values()
        if (s is not None) and (kblinks['class_id'] is not None)
    }

    kblinks['entity_prop_exists'] = {
        str(s): {
            str(p): bool(kb.o(s,p))
            for p in kblinks['properties_id'].values() if (p is not None)
        }
        for s in kblinks['entities_id'].values() if (s is not None)
    }
    
    kblinks['rownr_colnr_matches'] = {}
    for rownr, row in enumerate(table['rows']):
        e = kblinks['entities_id'].get(str(rownr), None)
        if e is not None:
            colnr_matches = {}
            for colnr, cell in enumerate(row):
                p = kblinks['properties_id'].get(str(colnr), None)
                if p is not None:
                    score, literal, dtype = kb.match(e, p, cell)
                    if score:
                        colnr_matches[str(colnr)] = {
                            'score': score,
                            'lit': literal,
                            'dtype': dtype,
                        }
            kblinks['rownr_colnr_matches'][str(rownr)] = colnr_matches
    
    return kblinks

def get_novelty(kb, kblinks):
    lbl_colnr = next((colnr for colnr, p in kblinks['properties_id'].items() if p == kb.p_label), None)
    novelty = {
        'lbl': sum((not ppe.get(str(kb.p_label))) for e,ppe in kblinks['entity_prop_exists'].items()),
        'lbl_nomatch': sum(not bool(cm.get(str(lbl_colnr), {})) for cm in kblinks['rownr_colnr_matches'].values()),
        'lbl_total': sum(1 for e,ppe in kblinks['entity_prop_exists'].items()),
        
        'cls': sum((not hc) for e,hc in kblinks['entity_hasclass'].items()),
        'cls_total': len(kblinks['entity_hasclass']),
        
        'prop': sum((not pe) for e,ppe in kblinks['entity_prop_exists'].items() 
                    for p,pe in ppe.items() if p != str(kb.p_label)),
        'prop_nomatch': sum(not cm.get(str(colnr)) for cm in kblinks['rownr_colnr_matches'].values()
                        for colnr in kblinks['properties_id'] if colnr != lbl_colnr ),
        'prop_total': sum(1 for e,ppe in kblinks['entity_prop_exists'].items() 
                          for colnr, p in kblinks['properties_id'].items() if p != kb.p_label),
    }
    novelty['lbl_nomatch'] = novelty['lbl_nomatch']-novelty['lbl']
    novelty['prop_nomatch'] = novelty['prop_nomatch']-novelty['prop']
    
    novelty.update({
        'lbl_redundant': novelty['lbl_total'] - novelty['lbl_nomatch'] - novelty['lbl'],
        'cls_redundant': novelty['cls_total'] - novelty['cls'],
        'prop_redundant': novelty['prop_total'] - novelty['prop_nomatch'] - novelty['prop'],
    })
    
    novelty.update({
        'lbl_pct': (novelty['lbl'] / novelty['lbl_total']) if novelty['lbl_total'] else 0,
        'cls_pct': (novelty['cls'] / novelty['cls_total']) if novelty['cls_total'] else 0,
        'prop_pct': (novelty['prop'] / novelty['prop_total']) if novelty['prop_total'] else 0,
    })
    novelty.update({
        'lbl_val_pct': (novelty['lbl_nomatch'] / novelty['lbl_total']) if novelty['lbl_total'] else 0,
        'prop_val_pct': (novelty['prop_nomatch'] / novelty['prop_total']) if novelty['prop_total'] else 0,
    })
    return novelty