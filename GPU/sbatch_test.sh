#!/bin/bash
#Script for the test: $1 (matrix), $2 (block size), $3 (type of sorting), $4 (solution)
OUTPUT=outputs/test-$1-$2-$3-$4.out
ERROR=errors/test-$1-$2-$3-$4.err
cat <<EOF > temp_job.sh
#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --nodelist=edu01
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=test
#SBATCH --output=$OUTPUT
#SBATCH --error=$ERROR

./bin/SpMV $1 $2
EOF

sbatch temp_job.sh $1 $2


