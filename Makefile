
.PHONY: clean clean_hasy clean_leafsnap hasy leafsnap ets


HASY_URL = "https://zenodo.org/record/259444/files/HASYv2.tar.bz2"
LEAF_URL = "http://leafsnap.com/static/dataset/leafsnap-dataset.tar"
ETS_URL  = "https://people.csail.mit.edu/davidam/MCE/data/ETS.tgz"

# Aliases
hasy := data/processed/hasy
leafsnap := data/processed/leafsnap
ets := data/processed/ets

all: hasy leafsnap

clean_hasy:
	rm -rf data/raw/hasy/
	rm -rf data/processed/hasy/

clean_leafsnap:
	rm -rf data/raw/leafsnap/
	rm -rf data/processed/leafsnap/

clean_ets:
	rm -rf data/raw/ets/
	rm -rf data/processed/ets/

clean:
	rm -rf data/raw/*
	rm -rf data/processed/*

$(hasy):
	mkdir -p data/raw/hasy data/processed/hasy
	wget --continue --tries 0 $(HASY_URL) -P data/raw/hasy
	tar -xvjf data/raw/hasy/HASYv2.tar.bz2 --directory data/raw/hasy/
	scripts/process_hasy.sh
	cp data/raw/hasy/symbols.csv data/processed/hasy/

$(leafsnap):
	mkdir -p data/raw/leafsnap# data/processed/leafsnap
	wget --continue --tries 0 $(LEAF_URL) -P data/raw/leafsnap
	tar -xvf data/raw/leafsnap/leafsnap-dataset.tar --directory data/raw/leafsnap/
	scripts/process_leafsnap.sh

$(ets):
	#mkdir -p data/processed/ets
	#pip install spacy
	#python -m spacy download en
	wget --user davidam --ask-password $(ETS_URL) -P data/raw/ets
	tar -xvf data/raw/ets/ETS.tgz --directory data/raw/ETS/
	mv data/raw/ets/ETS_*/* data/raw/ets/
	rm -r data/raw/ets/ETS_*/
	scripts/process_ets.sh



hasy: $(hasy)
leafsnap: $(leafsnap)
ets: $(ets)
