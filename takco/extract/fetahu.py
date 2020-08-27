import os, sys, pickle, csv, json


def convert_table(doc, wpname_id=None):
    wpname_id = wpname_id or {}
    pgTitle = doc.get("entity", "")
    if pgTitle not in wpname_id:
        return
    pgId = wpname_id[pgTitle]

    for sec in doc.get("sections", []):
        sectionTitle = sec.get("section", "")
        sectionTitle = "" if (sectionTitle == "MAIN_SECTION") else sectionTitle

        for table in sec.get("tables", []):

            tableCaption = table.get("caption", None)

            tbNr = table.get("id", None)

            tableHeaders = []
            for hrow in table.get("header", []):
                tableHeader = []
                for hcell in hrow["columns"]:
                    tableHeader.append(
                        dict(text=hcell.get("name", ""), surfaceLinks=[],)
                    )
                tableHeaders.append(tableHeader)

            tableData = []
            for brow in table.get("rows", []):
                data = []
                for bcell in brow["values"]:
                    text = bcell.get("value", "")
                    surfaceLinks = []
                    for sv in bcell.get("structured_values", []):
                        if sv["structured"] and sv["structured"] in wpname_id:
                            surfaceLinks.append(
                                dict(
                                    offset=text.index(sv["anchor"]),
                                    endOffset=len(sv["anchor"]),
                                    linkType="INTERNAL",
                                    target=dict(
                                        id=wpname_id[sv["structured"]],
                                        title=sv["structured"],
                                    ),
                                    surface=sv["anchor"],
                                )
                            )
                    if surfaceLinks:
                        data.append(dict(text=text, surfaceLinks=surfaceLinks))
                    else:
                        data.append(dict(text=text))
                tableData.append(data)

            numCols = len(list(zip(*tableData)))
            numDataRows = len(tableData)
            numHeaderRows = len(tableHeaders)
            numericColumns = []

            yield dict(
                _id=f"{pgId}-{tbNr}",
                pgId=pgId,
                tbNr=tbNr,
                numCols=numCols,
                numDataRows=numDataRows,
                numHeaderRows=numHeaderRows,
                numericColumns=numericColumns,
                pgTitle=pgTitle,
                sectionTitle=sectionTitle,
                tableCaption=tableCaption,
                tableData=tableData,
                tableHeaders=tableHeaders,
            )


if __name__ == "__main__":
    try:
        _, PAGEIDS = sys.argv

        wpname_id = {}
        if PAGEIDS.endswith("tsv"):
            # ~ 1m
            for ri, row in enumerate(csv.reader(open(PAGEIDS), quotechar="'")):
                try:
                    id, ns, name = row
                    name = name.replace("'", "").replace("\\", "'")
                    if ns == "0":
                        id = int(id)
                        wpname_id[name] = id
                except:
                    pass
        #             if not ri % 1000:
        #                 print(f'Processed {ri:10d} rows...', end='\r')
        #         print()
        elif PAGEIDS.endswith("pickle"):
            # ~ 10s
            wpname_id = pickle.load(open(PAGEIDS, "rb"))

        for i, line in enumerate(sys.stdin):
            doc = json.loads(line)

            for t in convert_table(doc, wpname_id=wpname_id):
                print(json.dumps(t))
    except Exception as e:
        print(__doc__)
        raise e
