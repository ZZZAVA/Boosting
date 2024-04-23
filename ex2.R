library(rpart) 
library(mlbench) 
library(adabag)

data(Glass) 

l <- length(Glass[,1]) 
sub <- sample(1:l,7*l/10)  
mfinal <- seq(1, 201, 40) 
maxdepth <- 5 

resList <- list()

for(i in 1:length(mfinal)){
  Glass.adaboost <- bagging(Type ~.,data=Glass[sub,], mfinal=mfinal[i], maxdepth=maxdepth)
  Glass.adaboost.pred <- predict.bagging(Glass.adaboost, newdata=Glass[-sub, ]) 
  resList[i] <- Glass.adaboost.pred$error
  print(i)
  print(Glass.adaboost.pred$error)
}

plot(mfinal, resList)