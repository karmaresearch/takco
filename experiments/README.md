# TAKCO: Table-driven KB Completer

This repository holds the code to our experiments for the paper "Extracting Novel Facts from Tables for Knowledge Graph Completion" at ISWC.

## How to run

### Modified T2KMatch
- install https://github.com/bennokr/T2KMatch
    ```
    mvn install -Dmaven.test.skip=true
    ln -s "$(realpath ../T2KMatch/target/t2kmatch-2.1-jar-with-dependencies.jar)" data_t2k/t2kmatch-2.1-jar-with-dependencies.jar
    ln -s "$(realpath ../T2KMatch/data/)" data_t2k/db
    ln -s "$(realpath ../T2KMatch/data/webtables)" data_t2k/v0/tables    

    mkdir v1; cd v1
    mkdir tables; curl http://webdatacommons.org/webtables/tables_instance.tar.gz | tar -xzf - -C tables
    wget http://webdatacommons.org/webtables/classes_instance.csv
    mkdir attributes_instance; curl http://webdatacommons.org/webtables/attributes_instance.tar.gz | tar -xzf - -C attributes_instance
    mkdir entities_instance; curl http://webdatacommons.org/webtables/entities_instance.tar.gz | tar -xzf - -C entities_instance

    mkdir v2; cd v2    
    curl http://webdatacommons.org/webtables/extended_instance_goldstandard.tar.gz | tar -xzf -
    ```

### Modified STI
- install https://github.com/bennokr/sti
    - install http://archive.apache.org/dist/lucene/solr/5.4.0/
    - make KB files:
        - surfaceFormsScores.tsv (uripart,surfaceform,score)
        - redirects_label.tsv (from,to)
        - tables.sorted.uniq.nt.bak.tsv (s,p,o)
    - run `load_solr_data.sh`
    - run on CSVs:
        ```
        python convert_tables_csv_sti.py ../novelty/data_t2k/v1ex/tables/ output/v1ex/
        java -cp sti-jar-builder/target/sti-1.0alpha-jar-with-dependencies.jar uk.ac.shef.dcs.sti.experiment.TableMinerPlusBatch ./input/t2d_v1ex/ ./output/t2d_v1ex/ config/sti.properties -Dlog4j.configuration=./config/log4j.properties
        python convert_output_sti_tables.py output/t2d_v1ex/ output/t2d_v1ex/
        ```

### This system
- install https://github.com/karmaresearch/trident
    - `cmake .. -DPYTHON=1`
    - `echo "$(realpath ./)" > $(realpath ~/venv/lib/python3.6/site-packages/trident.pth)`
- install this repo
    - make input data, e.g.
        ```
        ls data_t2k/db/dbpedia/*.csv | while read fname; do echo $fname; python make_triples_from_dbpedia_tables.py $fname > $fname.nt; done
        cat data_t2k/db/dbpedia/*.csv.nt | sort | uniq > data_t2k/db/dbpedia.nt
        ```
    - load into trident
    - make label index
        - ...
    - run `candidates.py`
        - ...
    - run `match.py`
        - ...
    - run `disambiguate.py`
        - ...
    - run `error-analysis.py`
        - ...
    - run `prop-analysis.py`
        - ...
    - make triples
        - ...
    - run `triple-analysis.py`
        - ...

#### Debugging
```
FLASK_ENV=development FLASK_APP=error-analysis-app.py flask run
```