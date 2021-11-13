#!/bin/bash

MIN=0
MAX=1555

PREFIX="ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/XML/"
# fill this in
DOWNLOAD_DIR=

prev_num="0000"
for i in $(seq $MIN 5 $MAX); do
    num=$(printf "%04d" $i)
    fn="Compound_${prev_num}00001_${num}00000.xml"
    prev_num=$num
    echo "getting" $fn
    if ! [[ -f $DOWNLOAD_DIR$fn ]]; then
        orig_dir=$(pwd)
        cd $DOWNLOAD_DIR
        wget "${PREFIX}${fn}.gz"
        wget "${PREFIX}${fn}.gz.md5"
        if md5sum -c ${fn}.gz.md5; then
            echo md5 passed
            rm ${fn}.gz.md5
            # pigz does multithreaded unzipping. If you don't have pigz,
            # you can use gunzip by uncommenting the line below
            #gunzip $fn
            pigz -d -p 8 $fn
        else
            echo md5 failed
        fi
        cd $orig_dir
    fi
    python extract_info.py $DOWNLOAD_DIR$fn "<PC-Compound>" Preferred 11 34 -26 Traditional 11 34 -26 "Canonical<" 11 34 -26 Mass 12 34 -26 Formula 11 34 -26 "Log P" 11 34 -26 >> ${DOWNLOAD_DIR}iupacs_properties.txt
    rm $DOWNLOAD_DIR$fn
done
