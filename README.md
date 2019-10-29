# TAKCO: Table-driven KB Completer

This repository holds the code to our experiments for the paper "Extracting Novel Facts from Tables for Knowledge Graph Completion" at ISWC. The code for the original experiments is in the `experiments` directory.

It also contains a cleaned-up version of our method for use in practice. This version uses a `sqlite3` label index that is easier to set up and tailor to your specific needs.

Both versions use the <a href="https://github.com/karmaresearch/trident">trident</a> library.

## Usage

First, load your KG into trident (for usage information, run `trident help`).

Then, create the label index in sqlite:
```
python make_db.py path_to_trident_dir labels.sqlite
```

Finally, get table-KG matches (for example, table.csv with keycol=0):
```
python match.py path_to_trident_dir labels.sqlite table.csv 0 rowmatches.csv colmatches.csv
```

## Cite

If you find this research relevant to your work, please cite:
```
@inproceedings{kruit2019extracting,
  title={Extracting Novel Facts from Tables for Knowledge Graph Completion},
  author={Kruit, Benno and Boncz, Peter and Urbani, Jacopo},
  booktitle={International Semantic Web Conference},
  pages={364--381},
  year={2019},
  organization={Springer}
}
```