#!/bin/bash

OUT_TMP=$2_tmp

# delete previous files
rm -f $2
rm -f $OUT_TMP

for i in $( ls $1 ); do
	cat $1/$i >> $OUT_TMP
	printf "\n##########\n" >> $OUT_TMP
done

cat $OUT_TMP | sed 's;http://.*;web.archive.org;' | sed 's;\(@entity[0-9]*\):.*;\1;' > $2

rm $OUT_TMP

