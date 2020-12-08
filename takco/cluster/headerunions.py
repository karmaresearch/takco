from takco import Table

def table_get_headerId(table):
    return Table(table).headerId

def combine_by_first_header(table1, table2):
    return Table(table1).append(Table(table2))