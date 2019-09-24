#!/bin/bash
echo "Opening pattern" $1
files="$@"
for file in $files
do 
echo "Find and replace in" $file
sed -i 's/},}/}/g' $file
sed -i 's/]}}]/}}]/g' $file
done