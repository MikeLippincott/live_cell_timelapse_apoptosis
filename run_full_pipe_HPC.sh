#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=36:00:00
#SBATCH --partition=amilan
#SBATCH --qos=long
#SBATCH --output=alpine_std_out_std_err_full_pipe-%j.out


# this is a gold standard reproducible pipeline for the analysis of Live-Cell timelapse imaging data
# this is also meant to be run on a HPC cluster
# specific for the HPC cluster running SLURM

# set the first two modules run status
DOWNLOAD_AND_PREPROCESS_DATA=False

echo "Running the full pipeline..."

if [ $DOWNLOAD_AND_PREPROCESS_DATA = True ]; then
    echo "Downloading data..."
    # TODO: download the data and complete the module for a later date

    echo "Data downloaded."

    echo "Pre-processing data..."
    cd 1.pre_process_data/
    jid1=$(sbatch --dependency=afterany:$jid0 run_preprocessing_HPC.sh)

    cd ../2.cellprofiler_ic_processing/
    jid2=$(sbatch --dependency=afterany:$jid1 run_ic_HPC.sh)

    cd ../3b.run_sam/
    jid3=$(sbatch  --dependency=afterany:$jid2 run_sam2_HPC.sh)

    cd ../4.cellprofiler_analysis/
    jid4=$(sbatch  --dependency=afterany:$jid3 run_cellprofiler_analysis_HPC.sh)

    cd ../5.process_CP_features/
    jid5=$(sbatch  --dependency=afterany:$jid4 processing_features_HPC.sh)

else
    echo "Skipping the downloading and pre-processing step..."


    cd 2.cellprofiler_ic_processing/
    jid2=$(sbatch run_ic_HPC.sh)

    cd ../3b.run_sam/
    jid3=$(sbatch  --dependency=afterany:$jid2 run_sam2_HPC.sh)

    #cd ../4.cellprofiler_analysis/
    #jid4=$(sbatch  --dependency=afterany:$jid3 run_cellprofiler_analysis_HPC.sh)

    #cd ../5.process_CP_features/
    #jid5=$(sbatch  --dependency=afterany:$jid4 processing_features_HPC.sh)
    
    squeue -u $USER -o "%.8A %.4C %.10m %.20E"
fi

echo "Full pipeline complete."
