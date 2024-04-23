library(rpart) 
library(mlbench) 
library(adabag)

data(Vehicle) 
l <- length(Vehicle[,1]) 
sub <- sample(1:l,7*l/10)  

mfinal <- seq(1, 301, 60) 

maxdepth <- 5 

resList <- list()

for(i in 1:length(mfinal)){
  Vehicle.adaboost <- boosting(Class ~.,data=Vehicle[sub,], mfinal=mfinal[i], maxdepth=maxdepth)
  Vehicle.adaboost.pred <- predict.boosting(Vehicle.adaboost, newdata=Vehicle[-sub, ]) 
  resList[i] <- Vehicle.adaboost.pred$error
  print(i)
  print(Vehicle.adaboost.pred$error)
}

plot(mfinal, resList)
