#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --partition=amilan
#SBATCH --qos=long
#SBATCH --output=alpine_std_out_std_err_full_pipe-%j.out


# this is a gold standard reproducible pipeline for the analysis of Live-Cell timelapse imaging data
# this is also meant to be run on a HPC cluster
# specific for the HPC cluster running SLURM

cd 2.cellprofiler_ic_processing/

jid1=$(sbatch run_ic_HPC.sh)




cd ../3b.run_sam/



jid2=$(sbatch  --dependency=afterany:$jid1 run_sam2_HPC.sh)

cd ../4.cellprofiler_analysis/

jid3=$(sbatch  --dependency=afterany:$jid2 run_cellprofiler_analysis_HPC.sh)


cd ../5.process_CP_features/

jid4=$(sbatch  --dependency=afterany:$jid3 processing_features_HPC.sh)

