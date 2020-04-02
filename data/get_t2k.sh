cd raw

mkdir t2d-v1; cd t2d-v1
mkdir tables; curl http://webdatacommons.org/webtables/tables_instance.tar.gz | tar -xzf - -C tables
wget http://webdatacommons.org/webtables/classes_instance.csv
mkdir attributes_instance; curl http://webdatacommons.org/webtables/attributes_instance.tar.gz | tar -xzf - -C attributes_instance
mkdir entities_instance; curl http://webdatacommons.org/webtables/entities_instance.tar.gz | tar -xzf - -C entities_instance
cd ..

mkdir t2d-v2; cd t2d-v2    
curl http://webdatacommons.org/webtables/extended_instance_goldstandard.tar.gz | tar -xzf -
cd ..
cd ..
