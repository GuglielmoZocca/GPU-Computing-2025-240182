#!/bin/bash

rm bin/SpMV
make "MACROS=-D dtype=float -D $1 -D $2 -D $3"

for (( i=32; i <= 1024; i=i*2 ))
do
sbatch sbatch_script.sh Cities.mtx $i
sbatch sbatch_script.sh BenElechi1.mtx $i
sbatch sbatch_script.sh degme.mtx $i
sbatch sbatch_script.sh Hardesty2.mtx $i
sbatch sbatch_script.sh rail4284.mtx $i
sbatch sbatch_script.sh specular.mtx $i
sbatch sbatch_script.sh t2em.mtx $i
sbatch sbatch_script.sh torso1.mtx $i
sbatch sbatch_script.sh mawi_201512020130.mtx $i
done

