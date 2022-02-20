start_time = Sys.time()

library(glmnet)
library(R.matlab)

# paths
# data_chunk_file = "/n/holystore01/LABS/uchida_users/Users/mcampbell/patchforaging_glm/data_chunks/data_chunks_redistributed_384.mat"
#
#
data_chunk_file = "C:/data/patch_foraging_neuropix/data_chunks/data_chunks_redistributed_384.mat"
data_folder = "C:/data/patch_foraging_neuropix/GLM_input/20210212_original_vars/for_R/"
save_folder = "C:/data/patch_foraging_neuropix/GLM_output/run_20210212_R_test/"
if(!dir.exists(save_folder)) { dir.create(save_folder) }

# which chunk of data to process in this job
chunk_idx = "1" # Sys.getenv("SLURM_ARRAY_TASK_ID")
chunk_idx = as.numeric(chunk_idx)
data_chunks = readMat(data_chunk_file)
cellID = data_chunks[[1]][[chunk_idx]][[1]]
session = data_chunks[[2]][[chunk_idx]][[1]]

# load data
cat("Loading data for chunk ",chunk_idx," (",session,")...\n",sep="")
dat = readMat(paste(data_folder,session,".mat",sep=""))
x = dat[[1]]
alpha = dat[[2]]
base_var = dat[[3]]
foldid = dat[[4]]
cellID_all = dat[[5]]
spikecounts = dat[[6]]
y_all = spikecounts[,is.element(cellID_all,cellID)] # extract spikecounts for cells in this job
cat("done. (Elapsed time = ",difftime(Sys.time(),start_time,units="sec")," sec)\n",sep="")

# iterate over cells
beta = matrix()
length(beta) = length(cellID) * (dim(x)[2]+1)
dim(beta) = c(length(cellID), dim(x)[2]+1)
dev = matrix()
length(dev) = length(cellID)
dim(dev) = c(length(cellID),1)
for (i in 1:length(cellID)) {
	cat("Fitting cell ",i,"/",length(cellID),": ",cellID[i],"...",sep="")
	y = y_all[,i]
	fit = cv.glmnet(x, y, family = "poisson", alpha = alpha, foldid = foldid)
	beta[i,] = matrix(coef(fit, s = "lambda.1se"))
	dev[i] = matrix(fit$cvm[fit$lambda==fit$lambda.1se]) # mean cross-validated deviance
	cat("done. (Elapsed time = ",difftime(Sys.time(),start_time,units="sec")," sec)\n",sep="")
}

# write data
cat("Writing output...")
writeMat(paste(save_folder,"chunk",sprintf("%03d",chunk_idx),".mat",sep=""),beta=beta,dev=dev,session=session,cellID=cellID)
cat("done. (Elapsed time = ",difftime(Sys.time(),start_time,units="sec")," sec)\n",sep="")