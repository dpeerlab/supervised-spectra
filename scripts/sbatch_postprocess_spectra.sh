#!/bin/bash

## update on 10/08/2024
# this was to include PBS cells and a fixed gene set dictionary


LOG_DIR='/data/peer/rsaha/supervised-spectra/scripts/logs'

# processing script
PROC_SCRIPT="/data/peer/rsaha/supervised-spectra/scripts/postprocess_spectra_v2.py"

## specify run parameters

## anndata objects
ADATA='/data/peer/rsaha/supervised-spectra/scripts/results/cui23_processed_spectrav2_example_20241216_spectra_epochs_10000-time_1740428884.7412221.h5ad'

## model dirs
# results are output here in the separate folders
# postprocessing script will output into same folder
# MODEL_DIR='/data/peer/sam/cytokine_central/models/cui23/full/v1_pbs'
MODEL_FILE="/data/peer/rsaha/supervised-spectra/scripts/results/cui23_processed_spectrav2_example_20241216_spectra_epochs_10000-time_1740428884.7412221.pickle" 
 

# gene set dictionaries
GS_DICT='/data/peer/rsaha/supervised-spectra/scripts/results/gene_dict.json'


# obs key that was used for spectra
OBS_KEY='cytokine_coarse'

RESULTS_DIR='/data/peer/rsaha/supervised-spectra/scripts/results/'

#### run postprocess ####

# DIRS=(${MODEL_DIR}/*)

echo "running postprocess_spectra for $i"
TMP=$(mktemp)
echo "#!/bin/bash" > ${TMP}
echo "#SBATCH -J spectra_cui_post" >> ${TMP}
echo "#SBATCH -o ${LOG_DIR}/spectra_cui_full_post_%j.out" >> ${TMP}
echo "#SBATCH -e ${LOG_DIR}/spectra_cui_full_post_%j.err" >> ${TMP}
echo "#SBATCH --partition=cpuqueue" >> ${TMP}
echo "#SBATCH -n 20 --ntasks-per-node=20 --mem-per-cpu=20G --time=72:00:00" >> ${TMP}

# MODEL_FILE=$(ls -1 ${i}/*.pickle | xargs -n 1 basename)
echo "source ~/.bashrc" >> ${TMP}
echo "cd /data/peer/rsaha/supervised-spectra" >> ${TMP}
echo "poetry shell" >> ${TMP}
echo "cd /data/peer/rsaha/supervised-spectra/scripts" >> ${TMP}

echo "poetry run python ${PROC_SCRIPT} ${ADATA} ${MODEL_FILE} None ${GS_DICT} ${RESULTS_DIR} ${OBS_KEY} None None None" >> ${TMP}

chmod +x ${TMP}
cat ${TMP}
sbatch < ${TMP}

