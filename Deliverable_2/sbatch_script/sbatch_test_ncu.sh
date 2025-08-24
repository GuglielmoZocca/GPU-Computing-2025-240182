#!/bin/bash
#Script for the ncu test: $1 (matrix), $2 (type of sorting), $3 (solution)

OUTPUT=outputs/test-$1-$2-$3.out
ERROR=errors/test-$1-$2-$3.err


cat <<EOF > temp_job.sh
#!/bin/bash
#SBATCH --partition=edu-medium
#SBATCH --nodelist=edu01
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=test
#SBATCH --output=$OUTPUT
#SBATCH --error=$ERROR

sudo /opt/shares/cuda/software/CUDA/12.3.2/bin/ncu --nvtx --set full -o test/report_ncu/report-$1-$2-$3 -f bin/SpMV $1.mtx
sudo /opt/shares/cuda/software/CUDA/12.3.2/bin/ncu --import test/report_ncu/report-$1-$2-$3.ncu-rep --csv > test/report_ncu/report-$1-$2-$3.csv
EOF


sbatch temp_job.sh $1 $2 $3


