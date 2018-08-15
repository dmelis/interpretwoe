
.PHONY: clean clean_hasy hasy ets leafsnap


HASY_URL = "https://zenodo.org/record/259444/files/HASYv2.tar.bz2"
LEAF_URL = "http://leafsnap.com/static/dataset/leafsnap-dataset.tar"

# Aliases
hasy := data/processed/hasy
ets := data/processed/hasy
leafsnap := data/processed/leafsnap

all: hasy ets leafsnap

clean_hasy:
	rm -rf data/raw/hasy/
	rm -rf data/processed/hasy/

clean_leafsnap:
	rm -rf data/raw/leafsnap/*
	rm -rf data/processed/leafsnap/*

clean:
	rm -rf data/raw/*
	rm -rf data/processed/*

$(hasy):
	mkdir -p data/raw/hasy data/processed/hasy
	wget $(HASY_URL) -P data/raw/hasy
	tar -xvjf data/raw/hasy/HASYv2.tar.bz2 --directory data/raw/hasy/
	scripts/process_hasy.sh
	cp data/raw/hasy/symbols.csv data/processed/hasy/

$(leafsnap):
	mkdir -p data/raw/leafsnap# data/processed/leafsnap
	wget $(LEAF_URL) -P data/raw/leafsnap
	tar -xvf data/raw/leafsnap/leafsnap-dataset.tar --directory data/raw/leafsnap/
	scripts/process_leafsnap.sh

hasy: $(hasy)

ets: $(ets)

leafsnap: $(leafsnap)
