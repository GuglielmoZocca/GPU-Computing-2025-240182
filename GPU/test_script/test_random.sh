#!/bin/bash

#!/bin/bash
rm bin/SpMV
make "MACROS=-D dtype=float -D $4 -D $5 -D $6 -D RAND"
for (( i=1; i <= $7; ++i ))
do
for (( j=32; j <= 1024; j = j*2 ))
do
sbatch sbatch_script_rand.sh $1 $2 $3 $j $i
done
done


