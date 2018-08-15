#!/bin/bash

DATADIR=./data/raw/leafsnap #~/.datasets/leafsnap-dataset
OUTDIR=./data/processed/leafsnap

TYPE=lab # field or lab
VERSION=4 # The photo version chosen - seems like 1-4 are same leaf in different light condiotions

PREFIX=$DATADIR/dataset/images/$TYPE

for dir in $PREFIX/* ; do
  #echo "$d"
  dirname=$(basename -- "$dir")
  echo $dirname
  mkdir -p $OUTDIR/$dirname
  cp $PREFIX/$dirname/*-$VERSION.jpg $OUTDIR/$dirname/
done
