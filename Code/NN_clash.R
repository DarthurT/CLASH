
#set working directory
path <- ""
setwd(path)

#load libraries and data
library(data.table)
library(mlr)
library(ggplot2)
library(purrr)
library(funModeling)
library(dplyr)

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
data1 <- subset(data1, select = -c(Name,Tp.toe,Hm0.toe, Tm.toe,Rc,q, RF,CF, X, X.1, X.2, Pow, Remark, Reference) ) #drop specific column

## Convert factor columns into numeric
data1[-1] <- lapply(data1[-1], function(x) as.numeric(as.character(x)))
## Convert factor columns into numeric
data1[1] <- lapply(data1[1], function(x) as.numeric(as.character(x)))

#data1$V30 <- as.numeric((data1$V30))
data2 <- data1[complete.cases(data1), ] #omit rows with NULL values in any row #use hold3 as hold 4 gets rid of too much

#data1[, c(1)] <- sapply(data1[, c(1)], as.factor)
#data1[, c(18:26,28,29)] <- sapply(data1[, c(18:26,28,29)], as.numeric)

#data1$TotalPE <- as.numeric((data1$TotalPE))
#data1$SiteCapacity <- as.numeric((data1$SiteCapacity))
#data1$SiteDryWeatherFlow <- as.numeric((data1$SiteDryWeatherFlow))
#data1$LiveStorage <- as.numeric((data1$LiveStorage))
#data1$AuthDailyAbstraction <- as.numeric((data1$AuthDailyAbstraction))
#data1$SSTStorageVol <- as.numeric((data1$SSTStorageVol))
#data1$BoreholeDepth <- as.numeric((data1$BoreholeDepth))
#data1$BorefoleDiameter <- as.numeric((data1$BorefoleDiameter))
#data1$PercentUtilised <- as.numeric((data1$PercentUtilised))
#data1$LastMajorReferb <- as.numeric((data1$LastMajorReferb))

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
parallelStartSocket(cpus = 6)
#parallelstart


#Random Forest 
rf.lrn <- makeLearner("regr.nnet")
#rf.lrn$par.vals <- list(ntree = 100L,
#                        importance=TRUE)

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
#scale_y_log10(limits = c(0.001,1e1)) +
#scale_x_log10(limits = c(0.001,1e1)) + 
  stat_smooth() +
  theme_bw()  + geom_abline(intercept = 0)
p + labs(x="Tm-1,0 pred (s)", y="Tm-1,0 actual (s)") 

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


parallelStartSocket(cpus = 6)

#parallelstart
getParamSet(rf.lrn)
#rf.lrn$par.vals <- list(ntree = 1000L,
#                        importance=TRUE)

#set parameter space
params <- makeParamSet(
  makeIntegerParam("size",lower = 1,upper = 30),
  makeIntegerParam("decay",lower = 1000, upper = 5000)
)

#set validation strategy
rdesc <- makeResampleDesc("CV",iters=25)

#set optimization technique
ctrl <- makeTuneControlRandom(maxit = 100L)

parallelStartSocket(cpus = detectCores())
tune <- tuneParams(learner = rf.lrn
                   ,task = testtask
                   ,resampling = rdesc
                   ,measures = list(arsq,mae,mape,rmse,timetrain)
                   ,par.set = params
                   ,control = ctrl
                   ,show.info = T)

#turn tuneparams as learner
lrn = setHyperPars(makeLearner("regr.nnet"), par.vals = tune$x)
tunedlrn = train(lrn, traintask)
results <- predict(tunedlrn, task = traintask)



#generate data on tuning effects
data = generateHyperParsEffectData(tune)

plotHyperParsEffect(data, x = "size", y = 'iteration', plot.type = "line",
                    partial.dep.learn = "regr.randomForest")

#plot heatmap of tuning effects parameters (nodesize and mtry)
plt = plotHyperParsEffect(data, x = "size", y = "decay", z = "mae.test.mean",plot.type = "heatmap", interpolate = "regr.nnet")
min_plt = min(data$data$adjrsq.test.mean, na.rm = TRUE)
max_plt = max(data$data$adjrsq.test.mean, na.rm = TRUE)
med_plt = mean(c(min_plt, max_plt))
plt + scale_fill_gradient2(breaks = seq(min_plt, max_plt, length.out = 20),
                           low = "green", mid = "orange", high = "red", midpoint = med_plt)
plt + labs(fill = "Test Score") 
plt +  theme_minimal() 


# extract actual and predicted
#measure_results <- as.data.frame(tunedlrn[['measures.test']])
pred <- (results[["data"]][["response"]])
true <- (results[["data"]][["truth"]])
pd <- rbind(pred,true)
pd <- t(pd)
pd <- as.data.frame(pd)



p <- ggplot(pd, aes(true, pred))  +
  geom_point() + 
 # scale_y_log10(limits = c(0.0000001,1e0)) +
 # scale_x_log10(limits = c(0.0000001,1e0)) + 
  xlim(0, 5) + 
  ylim(0, 5) +
  stat_smooth() +
  theme_bw() 
p + geom_abline(intercept = 0) #plot x=y line