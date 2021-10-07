
#set working directory
setwd("C:/Users/daniel.thompson/OneDrive - Swansea University/02_Sandbox/09_Waves/02_CLASH")

#load libraries and data
library(data.table)
library(mlr)
library(ggplot2)
library(purrr)
library(funModeling)
library(dplyr)
library(tidyverse)
library(lme4)

library(lme4)
library(mgcv)

# create remove outliers function - DT
remove_outliers <- function(x, na.rm = TRUE, ...) {
  qnt <- quantile(x, probs=c(.02, .98), na.rm = na.rm, ...)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  y
}


#load data
data1 <- read.csv("Database_20050101.csv", header = TRUE,
                  sep=",",na.strings = c(" NA"), stringsAsFactors=FALSE)
data1 = data1[-1,]
data1 <- subset(data1, select = -c(Name, RF,CF, X, X.1, X.2, Pow, Remark, Reference) ) #drop specific column

## Convert factor columns into numeric
data1[-1] <- lapply(data1[-1], function(x) as.numeric(as.character(x)))
## Convert factor columns into numeric
data1[1] <- lapply(data1[1], function(x) as.numeric(as.character(x)))

#data1$V30 <- as.numeric((data1$V30))
data2 <- data1[complete.cases(data1), ] #omit rows with NULL values in any row #use hold3 as hold 4 gets rid of too much


test <- data2 %>% group_by(Rc) %>% tally()

m <- lmer(q ~ Hm0.deep + Tp.deep + (1 | Rc ), data = data2)


m <- lmer(q ~ (1 | Rc ), data = data2)

summary(m)

m2 <- lm(q ~ 1, data = data2)

summary(m2)

2*(logLik(m) - logLik(m2))

#huge evidence of Rc on overtopping discharge. 
#need to think of a way to classify structure groups. 

#-----------------------------------------------------------------#
#---------------------# GLM #---------------------#
#-----------------------------------------------------------------#







#-----------------------------------------------------------------#
#---------------------# Sort and clean data #---------------------#
#-----------------------------------------------------------------#



trainnum = round(0.8*nrow(data2))-1 #80% of data used for training
testnum = round(0.2*nrow(data2))-1 #20% of data used for test
set.seed(101)                         #set seed number for random repeatability 
train <- data2[sample(nrow(data2), trainnum), ]
test <- data2[sample(nrow(data2), testnum), ]

#-----------------------------------------------------------------#
#-------------------------# Build learner #-----------------------#
#-----------------------------------------------------------------#

#use MLR
#create a task
traintask <- makeRegrTask(data = train,target = "Tm.1.0.toe")
testtask <- makeRegrTask(data = test,target = "Tm.1.0.toe")

#set 'n' number of fold cross validation (number of iterations, will 
#provide n number of rmse or r2 values - confirm what this value does Daniel)
rdesc <- makeResampleDesc("CV",iters=50L)

#set parallel backend
library(parallelMap)
library(parallel)
parallelStartSocket(cpus = 4)
#parallelstart


#Random Forest 
rf.lrn <- makeLearner("regr.randomForest")
rf.lrn$par.vals <- list(ntree = 1000L,
                        importance=TRUE)

r <- resample(learner = rf.lrn
              ,task = traintask
              ,resampling = rdesc
              ,measures = list(arsq,mae,mape,rmse,timetrain)
              ,show.info = T)


#-----------------------------------------------------------------#
#-------# create learning with plot of learning curve #-----------#
#-----------------------------------------------------------------#


### Learning curves plot datasize (percentage used for model)
### vs defined performance measure. 
rcurve = generateLearningCurveData(
  learners = rf.lrn,
  task = traintask,
  percs = seq(0.1, 1, by = 0.1),
  measures = arsq,
  resampling = rdesc,
  show.info = T)
plotLearningCurve(rcurve)

 # plot training and test mean

lc2 = generateLearningCurveData(learners = rf.lrn, task = traintask,
                                percs = seq(0.1, 1, by = 0.01),
                                measures = list(arsq, 
                                                setAggregation(arsq, train.mean)), 
                                resampling = rdesc,
                                show.info = T)
plotLearningCurve(lc2, facet = "learner")

# plot arsq and mean

lc3 = generateLearningCurveData(learners = rf.lrn, task = traintask,
                                percs = seq(0.1, 1, by = 0.05),
                                measures = list(arsq, rmse),
                                resampling = rdesc,
                                show.info = T)
plotLearningCurve(lc3)

#-----------------------------------------------------------------#
#-------------------- Plot prediction results --------------------#
#-----------------------------------------------------------------#

# extract actual and predicted
measure_results <- as.data.frame(r[['measures.test']])
pred <- (r[["pred"]][["data"]][["response"]])
true <- (r[["pred"]][["data"]][["truth"]])
pd <- rbind(pred,true)
pd <- t(pd)
pd <- as.data.frame(pd)


p <- ggplot(pd, aes(pred,true))  +
  geom_point() + 
 xlim(0, 12.0) + 
 ylim(0, 12.0) +
 #scale_y_log10(limits = c(0.00001,1e1)) +
 # scale_x_log10(limits = c(0.00001,1e1)) + 
  stat_smooth() +
  theme_bw() 
p + geom_abline(intercept = 0) #plot x=y line

#-----------------------------------------------------------------#
#---------------------- Test prediction --------------------------#
#-----------------------------------------------------------------#

n = getTaskSize(testtask)
train.set = seq(1, n, by = 2)
test.set = seq(2, n, by = 2)
lrn = rf.lrn
mod = train(lrn, testtask, subset = train.set)

task.pred = predict(mod, task = testtask, subset = test.set)
task.pred



#stop parallelization
parallelStop()

#-----------------------------------------------------------------#
#---------------------# Random Forest Tuning #--------------------#
#-----------------------------------------------------------------#

getParamSet(rf.lrn)
rf.lrn$par.vals <- list(ntree = 1000L,
                        importance=TRUE)

#set parameter space
params <- makeParamSet(
  makeIntegerParam("mtry",lower = 2,upper = 100),
  makeIntegerParam("nodesize",lower = 2,upper = 50)
)

#set validation strategy
rdesc <- makeResampleDesc("CV",iters=25)

#set optimization technique
ctrl <- makeTuneControlRandom(maxit = 1000L)

parallelStartSocket(cpus = 4)
tune <- tuneParams(learner = rf.lrn
                   ,task = testtask
                   ,resampling = rdesc
                   ,measures = list(arsq,rmse)
                   ,par.set = params
                   ,control = ctrl
                   ,show.info = T)

#turn tuneparams as learner
lrn = setHyperPars(makeLearner("regr.randomForest"), par.vals = tune$x)
tunedlrn = train(lrn, traintask)
results <- predict(tunedlrn, task = testtask)

# extract actual and predicted
#measure_results <- as.data.frame(tunedlrn[['measures.test']])
pred <- (results[["data"]][["response"]])
true <- (results[["data"]][["truth"]])
pd <- rbind(pred,true)
pd <- t(pd)
pd <- as.data.frame(pd)



p <- ggplot(pd, aes(true, pred))  +
  geom_point() + 
  scale_y_log10(limits = c(0.0000001,1e0)) +
  scale_x_log10(limits = c(0.0000001,1e0)) + 
 # xlim(0, 15000) + 
  #ylim(0, 15000) +
  stat_smooth() +
  theme_bw() 
p + geom_abline(intercept = 0) #plot x=y line