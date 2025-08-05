#!/bin/bash
#Script for the test: $1 (Matrix), $2 (SORT)
OUTPUT=outputs/test-$1-$2.out
ERROR=errors/test-$1-$2.err
#OUTPUTF="outputs/test-$1-$2.out"
cat <<EOF > temp_job.sh
#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --nodelist=edu01
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --job-name=test
#SBATCH --output=$OUTPUT
#SBATCH --error=$ERROR

./bin/SpMV $1
EOF

sbatch temp_job.sh $1