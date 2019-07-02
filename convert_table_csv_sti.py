import lxml.etree as ET
import csv
def csv_to_sti(in_fname, out_fname):
    
    # ASSUME header row
    theader, *trows = list(csv.reader(open(in_fname)))
    
    entity = ET.Element("entity")
    logicalTable = ET.SubElement(entity,  "logicalTable")
    content = ET.SubElement(logicalTable,  "content")
    header = ET.SubElement(content,  "header")
    for h in theader:
        cell = ET.SubElement(header,  "cell")
        ET.SubElement(cell,  "html").text = h
        ET.SubElement(cell,  "wikipedia")
    for r in trows:
        row = ET.SubElement(content,  "row")
        for c in r:
            cell = ET.SubElement(row,  "cell")
            ET.SubElement(cell,  "html").text = c
            ET.SubElement(cell,  "wikipedia")
    tableContext = ET.SubElement(logicalTable,  "tableContext")
    tree = ET.ElementTree(entity)
    tree.write(out_fname, pretty_print=True, xml_declaration=True, encoding='utf-8')

if __name__ == '__main__':
    import sys
    try:
        _, in_fname, out_fname = sys.argv
    except:
        print('Usage: convert_table_csv_sti.py in_file.csv out_file.xml')
        sys.exit(0)
    
    csv_to_sti(in_fname, out_fname)