library(rpart) 
library(mlbench) 
library(adabag)
library(kknn)

data(Vehicle) 

l <- length(Vehicle[,1]) 
sub <- sample(1:l,7*l/10)

Ttt.kknn <- kknn(Class ~ ., Vehicle[sub,], Vehicle[-sub,], k = floor(sqrt(7*l/10)))
fit <- fitted(Ttt.kknn)

tb <- table(fit,Vehicle$Class[-sub]) 
error.rpart <- 1-(sum(diag(tb))/sum(tb)) 
error.rpart
Vehicle.adaboost <- boosting(Class ~.,data=Vehicle[sub,], mfinal=25, maxdepth=5)
Vehicle.adaboost.pred <- predict.boosting(Vehicle.adaboost, newdata=Vehicle[-sub, ])
Vehicle.adaboost.pred[6]

