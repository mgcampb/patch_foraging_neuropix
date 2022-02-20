5/17/2021
Instructions for running GLM on RC cluster:
1) Generate regressor matrix for each session using "prepare_data_for_R_glmnet.m"
2) Move these .mat files into a folder in C:\data\patch_foraging_neuropix\GLM_output with the run name (e.g. 20210514_accel)
3) Move them onto the RC cluster using Globus, into a folder in holystore with the run name
4) Navigate to /n/holystore01/LABS/uchida_users/Users/mcampbell/run_GLM_on_cluster 
5) Edit the script "fit_data_chunk.R" to point to the folder with the regressor matrices
6) Run it using "sbatch --array=1-384 run_par_R.sh" (384 chunks of data are in the folder data_chunks - can edit this also)
7) Move the output (the "chunks" folder) into the folder in C:\data\...\GLM_output using Globus
8) Run "gather_chunks_by_session.m" to add the GLM output to the regressor matrix .mat files locally