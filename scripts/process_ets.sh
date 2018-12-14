#!/bin/bash

RESPDIR=./data/raw/ets/data/text/responses/tokenized

#awk '{print FILENAME}' $RESPDIR/*


IDXDIR=./data/raw/ets/data/text

# First, need to join files because index-test.csv doesnt have labels.
awk -F',' 'BEGIN{OFS="\t"; print "fid", "fold", "lang", "score";}
  FNR==NR{
    L[$1]=$3; S[$1]=$4;
    next;
  }{
    n = split(FILENAME, a, "/");
    fold = a[n];
    gsub("index-","",fold);
    gsub(".csv","",fold);
    gsub("ing","",fold);
    print $1, fold, L[$1], S[$1];
}' $IDXDIR/index.csv $IDXDIR/index-* > $IDXDIR/joint-index.tsv


OUTDIR=./data/processed/ets/
mkdir -p $OUTDIR
folds=("train" "dev" "test")
for f in "${folds[@]}"
do
  echo -e "fid\ttext\tlanguage\tscore" > $OUTDIR/$f.tsv
done

find $RESPDIR -name \*.txt |
xargs awk -v PREFIX="$OUTDIR" -F"\t" '
  BEGIN{
    OFS="\t";
  }
  FNR==NR{
  F[$1]=$2; L[$1]=$3; S[$1]=$4;
  next;
  }{
  n = split(FILENAME, a, "/");
  if(CURRF!=a[n] && CURRF!=""){
    OUTF=PREFIX F[CURRF] ".tsv";
    #print OUTF;
    print CURRF, TXT, L[CURRF], S[CURRF] > OUTF
    TXT=$0;
  } else {
    TXT=TXT" "$0;
  }
  CURRF=a[n];
  }' $IDXDIR/joint-index.tsv
