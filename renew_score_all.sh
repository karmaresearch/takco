
GOLD=$1
DIR=$2

ls $DIR/extracted_triples-*.csv | while read f; do
    echo $f;
    python renew_scores.py $GOLD $f;
done