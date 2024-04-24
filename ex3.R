library(rpart) 
library(mlbench) 
library(adabag)
library(kknn)


#change data and change Class to Type, ofc change args
data(Glass) 

l <- length(Glass[,1]) 
sub <- sample(1:l,7*l/10)

Ttt.kknn <- kknn(Type ~ ., Glass[sub,], Glass[-sub,], k = floor(sqrt(7*l/10)))
fit <- fitted(Ttt.kknn)

tb <- table(fit,Glass$Class[-sub]) 
error.rpart <- 1-(sum(diag(tb))/sum(tb)) 
error.rpart
Glass.adaboost <- boosting(Type ~.,data=Glass[sub,], mfinal=25, maxdepth=5)
Glass.adaboost.pred <- predict.boosting(Glass.adaboost, newdata=Glass[-sub, ])
Glass.adaboost.pred[6]

