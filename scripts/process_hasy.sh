#!/bin/bash

## Change these as desired
HASYDIR=./data/raw/hasy
OUTDIR=./data/processed/hasy

echo "Processing HASY dataset..."
while IFS=, read -r col1 col2 col3 col4
do
    if [ $col1 = "path" ]
    then
      continue
    fi
    echo "$col1 | $col2 | $col3 | $col4"
    mkdir -p "$OUTDIR/$col2"
    echo "$HASYDIR/$col1"
    cp "$HASYDIR/$col1" "$OUTDIR/$col2"
done <  $HASYDIR/hasy-data-labels.csv
