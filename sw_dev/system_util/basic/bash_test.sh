#!/usr/bin/env bash

# REF [site] >>
#	http://johnstowers.co.nz/pages/bash-cheat-sheet.html
#	https://bash.cyberciti.biz/guide/Main_Page

#------------------------------------------------------------

#[ $# -eq 0 ] && { echo "Usage: $0 file1 file2 ... fileN"; exit 1; }
[ $# -eq 0 ] && { echo "Usage: $0 file1 file2 ... fileN"; }

ARGS=$@
echo "args are $ARGS"

for f in $(ls /tmp/*)
do
	echo $f
done

for id in `seq 1 5`
do
	echo $id
done

#------------------------------------------------------------

VAR=2

if [ $VAR -eq 1 ]
then
	echo "VAR is 1".
elif [ $VAR -eq 2 ]
then
	echo "VAR is 2".
else
	echo "VAR is greater than 2".
fi

#------------------------------------------------------------

ARRAY_VAR=(3 6 9)

for id in 2 4 6 8 10
do
	echo "ID is ${id}".
done

for id in {1..5}
do
	echo "ID is ${id}".
done

for id in ${ARRAY_VAR[*]}
do
	echo "ID is ${id}".
done

for (( id=1; id<=5; ++id ))
do
	echo "ID is ${id}".
done

files="/etc/passwd /etc/group /etc/shadow /etc/gshdow"
for f in $files
do
	[ -f $f ] && echo "$f file found." || echo "$f file not found."
done
