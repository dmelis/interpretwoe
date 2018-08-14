#!/bin/bash

# Change these as desired
DATADIR=~/.datasets
HASYDIR=$DATADIR/HASYv2
OUTDIR=./data/Hasy

# Check if dataset has been downloaded already
if [ ! -d "$HASYDIR" ]; then
  echo "Downloading HASY dataset..."
  mkdir -p $HASYDIR
  wget https://zenodo.org/record/259444/files/HASYv2.tar.bz2 -P $HASYDIR
  tar -xvjf $HASYDIR/HASYv2.tar.bz2 --directory $HASYDIR
fi

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
