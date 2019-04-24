This repo contains an Rscript for plotting of pncA feature data (pncA_features) and for training machine learning models on
this data.

I recommend downloading and running in RStudio, as it has a nicer user interface and is easier to run individual parts of the
script to see how each step works. In addition, this will allow you to easily try different models and training parameters to
see if you can improve the model or use it for your own purposes.

The script makes use of multiple cores to speed up model training. If you have a Unix operating system, please modify
the code on line 30 (registerDoMC) of the script to reflect the number of cores you want to utilize. If on Windows, you will need to install the R package doParallel and snow and run the following code to register a local cluster on your machine.

#library(doParallel)
#library(snow)
#workers=makeCluster(4,type="SOCK")
#registerDoParallel(workers)

This script requires several packages that contain the code used to train the machine learning models,
so it will install several packages if they are not already present. The list of packages is at the beginning of the script.

Please contact carter_jjc@hotmail.com for any inquiries.


