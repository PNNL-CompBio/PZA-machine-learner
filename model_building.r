#This script fits machine learning models to an array containing structural features (continuous and factors)
# to model the probability of resistance/susceptibility for a particular missense mutation in pncA
# The script makes use of multiple cores to speed up model training. Please modify the code on line 14 to
# reflect the number of cores on your machine.

list.of.packages <- c("RColorBrewer", "pROC", "scatterplot3d", "ggplot2", "rgl", "caret", "doMC")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages,repos = "https://www.stats.bris.ac.uk/R/")

library(RColorBrewer)
library(pROC)
library(scatterplot3d)
library(rgl)
library(ggplot2)
library(caret)
library(doMC)
registerDoMC(cores=6)


#/////////////////////////// Loading and interrogating the data ////////////////////////////////////////////

#Read in data file, please modify file to point to a .csv of supplementary table X.
data <- read.csv(file = 'pncA_features', header = TRUE)
data$color <- 'red'
data[data$Phenotype=='R','color'] = 'blue'

#The following code was used to generate supplementary figure 1
ggplot(data, aes(x=Phenotype, y=MAPP, fill=color)) + geom_violin() + geom_jitter(shape=16, position=position_jitter(0.2)) + theme_classic()
wilcox.test(data$MAPP~data$Phenotype)

ggplot(data, aes(x=Phenotype, y=Distance_active, fill=color)) + geom_violin() + geom_jitter(shape=16, position=position_jitter(0.2)) + theme_classic()
wilcox.test(data$Distance_active~data$Phenotype)

ggplot(data, aes(x=Phenotype, y=rogov, fill=color)) + geom_violin() + geom_jitter(shape=16, position=position_jitter(0.2)) + theme_classic()
wilcox.test(data$rogov~data$Phenotype)

ggplot(data, aes(x=Phenotype, y=hbond, fill=color)) + geom_violin() + geom_jitter(shape=16, position=position_jitter(0.2)) + theme_classic()
wilcox.test(data$hbond~data$Phenotype)

ggplot(data, aes(x=Phenotype, y=abs_vol, fill=color)) + geom_violin() + geom_jitter(shape=16, position=position_jitter(0.2)) + theme_classic()
wilcox.test(data$abs_vol~data$Phenotype)

ggplot(data, aes(x=Phenotype, y=solvent2, fill=color)) + geom_violin() + geom_jitter(shape=16, position=position_jitter(0.2)) + theme_classic()
wilcox.test(data$solvent2~data$Phenotype)

ggplot(data, aes(x=Phenotype, y=consurf35, fill=color)) + geom_violin() + geom_jitter(shape=16, position=position_jitter(0.2)) + theme_classic()
wilcox.test(data$consurf35~data$Phenotype)

ggplot(data, aes(x=Phenotype, y=meta.ddG, fill=color)) + geom_violin() + geom_jitter(shape=16, position=position_jitter(0.2)) + theme_classic()
wilcox.test(data$meta.ddG~data$Phenotype)

wilcox.test(data$active~data$Phenotype)



#The following code generates plots of various features for visual inspection
plot(data$Distance_active,data$solvent2,col=data$color,pch=data$point,main='Distance vs Solvent',xlab="Distance to metal (Å)",ylab="% Solvent access")

plot(data$MAPP, data$meta.ddG, main="MAPP vs Meta-ddG",col=data$color,pch=data$point,xlab= "MAPP",ylab="Free energy of unfolding")

plot(data$consurf35, data$rogov, main="Rogov vs Consurf",col=data$color,pch=data$point,xlab= "Consurf",ylab="Rogov")

plot(data$DELTA_hydro,data$meta.ddG,main="Free energy change vs. absolute change in hydrophobicity",col=data$color,pch=data$point,xlab="Change in hydrophobicity",ylab="Free energy change (kcal/mol)")

plot(data$solvent2,data$meta.ddG,main="Free energy change vs Solvent Accessibility",col=data$color,pch=data$point,xlab="solvent access",ylab="Free energy change (kcal/mol)")

plot(data$solvent2,data$DELTA_hydro,main="Hydrophobic change vs Solvent Accessibility",col=data$color,pch=data$point,xlab="Core -------------> Surface",ylab="More hydrophobic ----------> More polar")

plot(data$solvent2,data$MAPP,main="MAPP vs Solvent Accessibility",col=data$color,pch=data$point,xlab="Core -------------> Surface",ylab="MAPP Score")

plot(data$DELTA_vol,data$solvent2,col=data$color,pch=data$point,xlab="Absolute change in volume",ylab="Solvent Access")

scatterplot3d(data$solvent2, data$MAPP,data$meta.ddG,main="PoPCORN Predictors",color=data$color,pch=data$point,xlab="Solvent accessibility",ylab="MAPP",zlab="Destabilization Factor")

plot3d(data$solvent2, data$MAPP,data$meta.ddG,main="PoPCORN Predictors",col=data$color,pch=data$point)






#///////////////////////////  Fitting the model   ////////////////////////////////////////

#Set data types for features
data$solvent2 <- as.numeric(data$solvent2)
data$meta.ddG <- as.numeric(data$meta.ddG)
data$MAPP <- as.numeric(data$MAPP)
data$DELTA_hydro <- as.numeric(data$DELTA_hydro)
data$abs_hydro <- as.numeric(data$abs_hydro)
data$hbond <- as.numeric(data$hbond)
data$hbond2 <- as.factor(data$hbond2)
data$rogov <- as.numeric(data$rogov)
data$consurf35 <- as.numeric(data$consurf35)
data$isSS <- as.factor(data$isSS)
data$destab <- as.factor(data$destab)
data$five <- as.factor(data$five)
data$active <- as.factor(data$active)
data$DELTA_vol <- as.numeric(data$DELTA_vol)
data$abs_vol <- as.numeric(data$abs_vol)
data$Distance_active <- as.numeric(data$Distance_active)
data$Phenotype <- as.factor(data$Phenotype)

#Remove absolute hydropathy change and absolute volume change
clean_data <- data[,c(1,2,3,4,6,8,9,10,11,12,13,14,15,16,17)]

#Set a seed for repeatability and split dataset into 70/30 training/testing sets.
set.seed(1234)
intrain <- createDataPartition(y = clean_data$Phenotype, p= 0.7, list = FALSE)
training <- clean_data[intrain,]
testing <- clean_data[-intrain,]

#Run backwards selection for logistic regression over the whole dataset to identify significant predictors
Resistance_Predictor <- glm(Phenotype ~ ., family=binomial(), data=clean_data)
backwards = step(Resistance_Predictor)
summary(backwards)

#Generate class weights to compensate for data imbalance
training_weights <- ifelse(training$Phenotype == "R",
                           (1/table(training$Phenotype)[1]) * 0.5,
                           (1/table(training$Phenotype)[2]) * 0.5)

#Train logisitic regression model and save calls.
set.seed(1234)
glm_tc <- trainControl(method="repeatedcv", number=10, repeats=10, search = "random", classProbs = TRUE, savePredictions = "final", summaryFunction = twoClassSummary, allowParallel = TRUE)
glm_fit <- train(Phenotype ~ Distance_active + MAPP + hbond + consurf35 + solvent2 + destab + MAPP*Distance_active + hbond*Distance_active + destab*hbond + destab*solvent2, data=training, method="bayesglm", family="binomial", metric='ROC', tuneLength = 200, trControl=glm_tc)
glm_fitpred <-  predict(glm_fit, newdata = training, type="prob")
glm_fitpred <- unlist(glm_fitpred[2])
glm_fitpred_t <- function(t,t2) ifelse(glm_fitpred<t, 0,ifelse(glm_fitpred<t2, 0.5, 1))
glm_filtered_calls <- glm_fitpred_t(0.4, 0.6)
train_truth <- factor(ifelse(training$Phenotype=="R", 0, 1), levels=c("0","1"))
glm_combine <- as.data.frame(glm_filtered_calls)
glm_combine$train_truth <- train_truth
glm_combine <- glm_combine[glm_combine$glm_filtered_calls!=0.5,]
glm_combine$glm_filtered_calls <- factor(glm_combine$glm_filtered_calls, levels=c("0",'1'))


#Apply logisitic regression model to testing data.
glm_testpred <- predict(glm_fit, newdata = testing, type="prob")
glm_testpred <- unlist(glm_testpred[2])
glm_testpred_t <- function(t,t2) ifelse(glm_testpred<t, 0,ifelse(glm_testpred<t2, 0.5, 1))
glm_test_filtered_calls <- glm_testpred_t(0.4, 0.6)
test_truth <- factor(ifelse(testing$Phenotype=="R", 0, 1), levels=c("0","1"))
glm_test_combine <- as.data.frame(glm_test_filtered_calls)
glm_test_combine$test_truth <- test_truth
glm_test_combine <- glm_test_combine[glm_test_combine$glm_test_filtered_calls!=0.5,]
glm_test_combine$glm_test_filtered_calls <- factor(glm_test_combine$glm_test_filtered_calls, levels=c("0",'1'))

#Center and scale features for use with support vector machine and neural network models
preProcValues <- preProcess(training, method = c("center", "scale"))
trainTransformed <- predict(preProcValues, training)
testTransformed <- predict(preProcValues, testing)


##################   SVM MODEL TRAINING   #####################################################
#Set seed and training controls
set.seed(1234)
svm_tc <- trainControl(method="repeatedcv", number=10, repeats=10, search = "random", classProbs = TRUE, savePredictions="final", summaryFunction = twoClassSummary, allowParallel = TRUE)

#Train svm with linear, radial, and polynomial kernels
svm_linear <- train(Phenotype ~ ., data=training, method="svmLinear", trControl=svm_tc, tuneLength = 200, metric = 'ROC', weights = training_weights)
svm_radial <- train(Phenotype ~ ., data=training, method="svmRadial", trControl=svm_tc, tuneLength = 200, metric = 'ROC', weights = training_weights)
svm_poly <- train(Phenotype ~ ., data=training, method="svmPoly", trControl=svm_tc, tuneLength = 200, metric = 'ROC', weights = training_weights)

#Generate predictions for test sets
svmL_testpred <- predict(svm_linear, newdata = testTransformed, type="prob")
svmL_testpred <- unlist(svmL_testpred[2])
svmR_testpred <- predict(svm_radial, newdata = testTransformed, type="prob")
svmR_testpred <- unlist(svmR_testpred[2])
svmP_testpred <- predict(svm_poly, newdata = testTransformed, type="prob")
svmP_testpred <- unlist(svmP_testpred[2])

#Draw ROCs for test sets
test_truth <- factor(ifelse(testTransformed$Phenotype=="R", 0, 1))
svm_test_ROC <- roc(test_truth ~ svmL_testpred, percent=TRUE)
plot(svm_test_ROC, col="red", print.auc=TRUE, print.auc.y=50)
svm_test_ROC <- roc(test_truth ~ svmR_testpred, percent=TRUE)
plot(svm_test_ROC, col="blue", print.auc=TRUE, add=TRUE, print.auc.y=45)
svm_test_ROC <- roc(test_truth ~ svmP_testpred, percent=TRUE)
plot(svm_test_ROC, col="green", print.auc=TRUE, add=TRUE, print.auc.y=40)

###############   select best model  ##############################################

#Save best svm models performance on training set
svm_fitpred <-  unlist(predict(svm_radial, newdata=trainTransformed, type="prob")[2])
svm_fitpred_t <- function(t,t2) ifelse(svm_fitpred<t, 0,ifelse(svm_fitpred<t2, 0.5, 1))
svm_filtered_calls <- svm_fitpred_t(0.4, 0.6)
svm_combine <- as.data.frame(svm_filtered_calls)
train_truth <- factor(ifelse(trainTransformed$Phenotype=="R", 0, 1))
svm_combine$train_truth <- train_truth
svm_combine <- svm_combine[svm_combine$svm_filtered_calls!=0.5,]
svm_combine$svm_filtered_calls <- factor(svm_combine$svm_filtered_calls, levels=c("0",'1'))

#Apply best svm model to test set
svm_testpred <- predict(svm_radial, newdata = testTransformed, type="prob")
svm_testpred <- unlist(svm_testpred[2])
svm_testpred_t <- function(t,t2) ifelse(svm_testpred<t, 0,ifelse(svm_testpred<t2, 0.5, 1))
svm_test_filtered_calls <- svm_testpred_t(0.4, 0.6)
svm_test_combine <- as.data.frame(svm_test_filtered_calls)
svm_test_combine$test_truth <- test_truth
svm_test_combine <- svm_test_combine[svm_test_combine$svm_test_filtered_calls!=0.5,]
svm_test_combine$svm_test_filtered_calls <- factor(svm_test_combine$svm_test_filtered_calls, levels=c("0",'1'))


################### Neural Network Model #########################################
#Train neural network models with and without weight decay
set.seed(1234)
nn_tc <- trainControl(method="repeatedcv", number=10, repeats=10, search='random', classProbs = TRUE, savePredictions="final", summaryFunction = twoClassSummary, allowParallel = TRUE)
nn_fit <- train(Phenotype ~ ., data=trainTransformed, method = 'mlpWeightDecay', trControl = nn_tc, tuneLength = 200, metric = 'ROC', weights = training_weights)
nn2_fit <- train(Phenotype ~ ., data=trainTransformed, method = 'avNNet', trControl = nn_tc, tuneLength = 200, metric = 'ROC', weights = training_weights)

#Generate predictions for test sets
nn_testpred <- predict(nn_fit, newdata = testTransformed, type="prob")
nn_testpred <- unlist(nn_testpred[2])
nn2_testpred <- predict(nn2_fit, newdata = testTransformed, type="prob")
nn2_testpred <- unlist(nn2_testpred[2])

#Draw ROCs for test sets
test_truth <- factor(ifelse(testTransformed$Phenotype=="R", 0, 1))
nn_test_ROC <- roc(test_truth ~ nn_testpred, percent=TRUE)
plot(nn_test_ROC, col="red", print.auc=TRUE, print.auc.y=50)
nn_test_ROC <- roc(test_truth ~ nn2_testpred, percent=TRUE)
plot(nn_test_ROC, col="blue", print.auc=TRUE, add=TRUE, print.auc.y=45)

####################### select best model ##############################

#Get training predictions for best model
nn_fitpred <- unlist(predict(nn_fit, newdata = trainTransformed, type="prob")[2])
nn_fitpred_t <- function(t,t2) ifelse(nn_fitpred<t, 0,ifelse(nn_fitpred<t2, 0.5, 1))
nn_filtered_calls <- nn_fitpred_t(0.4, 0.6)
nn_combine <- as.data.frame(nn_filtered_calls)
nn_combine$train_truth <- train_truth
nn_combine <- nn_combine[nn_combine$nn_filtered_calls!=0.5,]
nn_combine$nn_filtered_calls <- factor(nn_combine$nn_filtered_calls, levels=c("0",'1'))

#Apply best neural network model to test set
nn_testpred <- predict(nn_fit, newdata = testTransformed, type="prob")
nn_testpred <- unlist(nn_testpred[2])
nn_testpred_t <- function(t,t2) ifelse(nn_testpred<t, 0,ifelse(nn_testpred<t2, 0.5, 1))
nn_test_filtered_calls <- nn_testpred_t(0.4, 0.6)
nn_test_combine <- as.data.frame(nn_test_filtered_calls)
nn_test_combine$test_truth <- test_truth
nn_test_combine <- nn_test_combine[nn_test_combine$nn_test_filtered_calls!=0.5,]
nn_test_combine$nn_test_filtered_calls <- factor(nn_test_combine$nn_test_filtered_calls, levels=c("0",'1'))


#Generate truth tables for LR (train, test)
confusionMatrix(data=glm_combine$glm_filtered_calls, reference=glm_combine$train_truth, dnn=c('Predicted','Truth'), positive="0")
confusionMatrix(data=as.factor(glm_test_combine$glm_test_filtered_calls), reference=as.factor(glm_test_combine$test_truth), dnn=c('Predicted','Truth'), positive="0")
#Generate truth tables for SVM (train, test)
confusionMatrix(data=svm_combine$svm_filtered_calls, reference=svm_combine$train_truth, dnn=c('Predicted','Truth'), positive="0")
confusionMatrix(data=svm_test_combine$svm_test_filtered_calls, reference=svm_test_combine$test_truth, dnn=c('Predicted','Truth'), positive="0")
#Generate truth tables for NN (train, test)
confusionMatrix(data=nn_combine$nn_filtered_calls, reference=nn_combine$train_truth, dnn=c('Predicted','Truth'), positive="0")
confusionMatrix(data=nn_test_combine$nn_test_filtered_calls, reference=nn_test_combine$test_truth, dnn=c('Predicted','Truth'), positive="0")

#Pool calls for overall performance (reported with U calls in Figure 3 of paper)
glm_truth <- c(glm_combine$train_truth,glm_test_combine$test_truth)
glm_calls <- c(glm_combine$glm_filtered_calls,glm_test_combine$glm_test_filtered_calls)
svm_truth <- c(svm_combine$train_truth,svm_test_combine$test_truth)
svm_calls <- c(svm_combine$svm_filtered_calls,svm_test_combine$svm_test_filtered_calls)
nn_truth <- c(nn_combine$train_truth,nn_test_combine$test_truth)
nn_calls <- c(nn_combine$nn_filtered_calls,nn_test_combine$nn_test_filtered_calls)
#LR
confusionMatrix(data=as.factor(glm_calls), reference=as.factor(glm_truth), dnn=c('Predicted','Truth'), positive="1")
#SVM
confusionMatrix(data=as.factor(svm_calls), reference=as.factor(svm_truth), dnn=c('Predicted','Truth'), positive="1")
#NN
confusionMatrix(data=as.factor(nn_calls), reference=as.factor(nn_truth), dnn=c('Predicted','Truth'), positive="1")

#Plot model prediction histograms (Figure S2)
tmp <- training
tmp$preds <- unlist(predict(glm_fit, newdata = training, type="prob")[2]) #modify for desired model
hist(tmp[tmp$Phenotype=='R','preds'], col="#ff000020", breaks=20, xlim=c(0,1), freq=FALSE, xlab="Classifier Output (0=R, 1=S)", main="Distribution of GLM Classifier Output")
hist(tmp[tmp$Phenotype=='S','preds'], col="#00000050", breaks=20, xlim=c(0,1), freq=FALSE, add=T)

tmp <- trainTransformed
tmp$preds <- unlist(predict(svm_radial, newdata = trainTransformed, type="prob")[2]) #modify for desired model
hist(tmp[tmp$Phenotype=='R','preds'], col="#ff000020", breaks=20, xlim=c(0,1), freq=FALSE, xlab="Classifier Output (0=R, 1=S)", main="Distribution of SVM Classifier Output")
hist(tmp[tmp$Phenotype=='S','preds'], col="#00000050", breaks=20, xlim=c(0,1), freq=FALSE, add=T)

tmp <- trainTransformed
tmp$preds <- unlist(predict(nn_fit, newdata = trainTransformed, type="prob")[2]) #modify for desired model
hist(tmp[tmp$Phenotype=='R','preds'], col="#ff000020", breaks=20, xlim=c(0,1), freq=FALSE, xlab="Classifier Output (0=R, 1=S)", main="Distribution of NN Classifier Output")
hist(tmp[tmp$Phenotype=='S','preds'], col="#00000050", breaks=20, xlim=c(0,1), freq=FALSE, add=T)

#Plot roc curves (Figure 2)
train_truth <- factor(ifelse(training$Phenotype=="R", 0, 1))
test_truth <- factor(ifelse(testing$Phenotype=="R", 0, 1))
glm_train_ROC <- roc(train_truth ~ glm_fitpred, percent=TRUE)
glm_train_ci <- ci(glm_train_ROC, of = "se", sp = seq(0, 100, 5), boot.n=10000)
plot(glm_train_ROC, col="#FB9A99", print.auc=TRUE, main="Performance of GLM model on training set")
plot(glm_train_ci, type="shape", col=NA, border="#FB9A99", lty=2, no.roc=TRUE)
glm_test_ROC <- roc(test_truth ~ glm_testpred, percent=TRUE)
glm_test_ci <- ci(glm_test_ROC, of = "se", sp = seq(0, 100, 5), boot.n=10000)
plot(glm_test_ROC, col="#E31A1C", print.auc=TRUE, add=TRUE, print.auc.y=45)
plot(glm_test_ci, type="shape", col=NA, border="#E31A1C", lty=2, no.roc=TRUE)

svm_train_ROC <- roc(train_truth ~ svm_fitpred, percent=TRUE)
svm_train_ci <- ci(svm_train_ROC, of = "se", sp = seq(0, 100, 5), boot.n=10000)
plot(svm_train_ROC, col="#A6CEE3", print.auc=TRUE, main="Performance of SVM model on training set")
plot(svm_train_ci, type="shape", col=NA, border="#A6CEE3", lty=2,  no.roc=TRUE)
svm_test_ROC <- roc(test_truth ~ svm_testpred, percent=TRUE)
svm_test_ci <- ci(svm_test_ROC, of = "se", sp = seq(0, 100, 5), boot.n=10000)
plot(svm_test_ROC, col="#1F78B4", print.auc=TRUE, add=TRUE, print.auc.y=45)
plot(svm_test_ci, type="shape", col=NA, border="#1F78B4", lty=2, no.roc=TRUE)

nn_train_ROC <- roc(train_truth ~ nn_fitpred, percent=TRUE)
nn_train_ci <- ci(nn_train_ROC, of = "se", sp = seq(0, 100, 5), boot.n=10000)
plot(nn_train_ROC, col="#B2DF8A", print.auc=TRUE, main="Performance of NN model on training set")
plot(nn_train_ci, type="shape", col=NA, border="#B2DF8A", lty=2,  no.roc=TRUE)
nn_test_ROC <- roc(test_truth ~ nn_testpred, percent=TRUE)
nn_test_ci <- ci(nn_test_ROC, of = "se", sp = seq(0, 100, 5), boot.n=10000)
plot(nn_test_ROC, col="#33A02C", print.auc=TRUE, add=TRUE, print.auc.y=45)
plot(nn_test_ci, type="shape", col=NA, border="#33A02C", lty=2, no.roc=TRUE)

#Let's examine errors on training/testing sets
train_truth <- ifelse(training$Phenotype=="R", 0, 1)
test_truth <- ifelse(testing$Phenotype=="R", 0, 1)
training$diffs <- glm_filtered_calls - train_truth
training$preds <- glm_fitpred
testing$preds <- glm_testpred
testing$diffs <- glm_test_filtered_calls - test_truth
glm_diffs <- rbind(training, testing)
glm_diffs$type = 'glm'

training$diffs <- svm_filtered_calls - train_truth
testing$diffs <- svm_test_filtered_calls - test_truth
svm_diffs <- rbind(training, testing)
svm_diffs$type = 'svm'

training$diffs <- nn_filtered_calls - train_truth
testing$diffs <- nn_test_filtered_calls - test_truth
nn_diffs <- rbind(training, testing)
nn_diffs$type = 'nn'

diffs <- rbind(glm_diffs,svm_diffs,nn_diffs)

#Create diffs column where -1 is major error, -0.5 is minor error, 1 is very major error, 2 is correct R, and 3 is correct S
diffs <- diffs[!duplicated(diffs[,c(1:13)]),]
diffs[diffs$diffs==0.5,'diffs'] = -0.5
diffs[diffs$diffs==0 & diffs$Phenotype=='R','diffs'] = 2
diffs[diffs$diffs==0 & diffs$Phenotype=='S','diffs'] = 3
diffs$diffs <- as.factor(diffs$diffs)

#Create plots from Figure 4 of the paper
ggplot(diffs, aes(x=diffs, y=MAPP, fill=diffs)) + scale_fill_manual(values=c("pink","#bababa","magenta","#ca0020","#0571b0")) + geom_violin(show.legend=FALSE) + geom_jitter(shape=16, position=position_jitter(0.2), show.legend = FALSE) + theme_classic()

ggplot(diffs, aes(x=diffs, y=consurf35, fill=diffs)) + scale_fill_manual(values=c("pink","#bababa","magenta","#ca0020","#0571b0")) + geom_violin(show.legend=FALSE) + geom_jitter(shape=16, position=position_jitter(0.2), show.legend = FALSE) + theme_classic()

ggplot(diffs, aes(x=diffs, y=solvent2, fill=diffs)) + scale_fill_manual(values=c("pink","#bababa","magenta","#ca0020","#0571b0")) + geom_violin(show.legend=FALSE) + geom_jitter(shape=16, position=position_jitter(0.2), show.legend = FALSE) + theme_classic()

ggplot(diffs, aes(x=diffs, y=Distance_active, fill=diffs)) + scale_fill_manual(values=c("pink","#bababa","magenta","#ca0020","#0571b0")) + geom_violin(show.legend=FALSE) + geom_jitter(shape=16, position=position_jitter(0.2), show.legend = FALSE) + theme_classic()

ggplot(diffs, aes(x=diffs, y=meta.ddG, fill=diffs)) + scale_fill_manual(values=c("pink","#bababa","magenta","#ca0020","#0571b0")) + geom_violin(show.legend=FALSE) + geom_jitter(shape=16, position=position_jitter(0.2), show.legend = FALSE) + theme_classic()

ggplot(diffs, aes(x=diffs, y=hbond, fill=diffs)) + scale_fill_manual(values=c("pink","#bababa","magenta","#ca0020","#0571b0")) + geom_violin(show.legend=FALSE) + geom_jitter(shape=16, position=position_jitter(0.2), show.legend = FALSE) + theme_classic()

plot(diffs$diffs,diffs$destab)

#Run univariable logistic regression on the whole dataset to determine the performance of individual features
for(i in 2:17)
{univar_reg <- glm(Phenotype ~ data[,i], family=binomial(), data=clean_data)
roc <- roc(univar_reg$y ~ univar_reg$fitted.values, percent=TRUE)
print(roc$auc)}
