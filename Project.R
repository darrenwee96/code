
##Data contained:
  
##GRE Scores ( out of 340 )
##TOEFL Scores ( out of 120 )
##University Rating ( out of 5 )
##Statement of Purpose and
##Letter of Recommendation Strength ( out of 5 )
##Undergraduate GPA ( out of 10 )
##Research Experience ( either 0 or 1 )
##Chance of Admit ( ranging from 0 to 1 )

## Importing libraries
library(ggplot2)
library(ISLR)
library(fitdistrplus)
library(leaps)
library(glmnet)
library(corrplot)
library(gridExtra)
RNGkind(sample.kind = "Rounding")
set.seed(123)

## Importing csv
admissions <- read.csv("D:/Desktop/SMU/DSA211 Project final/Admission_Predict.csv")
View(admissions)

## Looking at the dataset
summary(admissions)

##checking for null values
sum(is.na(admissions))

##assigning columns to variable names
GRE <- admissions$GRE.Score
TOEFL <- admissions$TOEFL.Score
Ratings <- admissions$University.Rating
SOP <- admissions$SOP
LOR <- admissions$LOR
CGPA <- admissions$CGPA
Research <- admissions$Research
Chance <- admissions$Chance.of.Admit

##plotting correlation
numericVars <- which(sapply(admissions, is.numeric)) 
numericVarNames <- names(numericVars)
cat('There are', length(numericVars), 'numeric variables')
all_numVar <- admissions[, numericVars]
cor_numVar <- cor(all_numVar, use="pairwise.complete.obs") #correlations of all numeric variables

#sort on decreasing correlations
cor_sorted <- as.matrix(sort(cor_numVar[,'Chance.of.Admit'], decreasing = TRUE))
#select only high corelations
CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x)>0)))
cor_numVar <- cor_numVar[CorHigh, CorHigh]

corrplot.mixed(cor_numVar, tl.col="black", tl.pos = "lt")


##plotting Chance of admission against individual variables
p1 <- ggplot(data=admissions, aes(x=CGPA, y=Chance))+
  geom_point(col='blue') + geom_smooth(method = "lm", se=FALSE, color="black")
p2 <- ggplot(data=admissions, aes(x=GRE, y=Chance))+
  geom_point(col='red') + geom_smooth(method = "lm", se=FALSE, color="black")
p3 <- ggplot(data=admissions, aes(x=TOEFL, y=Chance))+
  geom_point(col='black') + geom_smooth(method = "lm", se=FALSE, color="black")
p4 <- ggplot(data=admissions, aes(x=Ratings, y=Chance))+
  geom_point(col='green') + geom_smooth(method = "lm", se=FALSE, color="black")
p5 <- ggplot(data=admissions, aes(x=SOP, y=Chance))+
  geom_point(col='yellow') + geom_smooth(method = "lm", se=FALSE, color="black")
p6 <- ggplot(data=admissions, aes(x=LOR, y=Chance))+
  geom_point(col='cyan') + geom_smooth(method = "lm", se=FALSE, color="black")
p7 <- ggplot(data=admissions, aes(x=admissions$Serial.No., y=Chance))+
  geom_point(col='tomato') + geom_smooth(method = "lm", se=FALSE, color="black")
grid.arrange(p1, p2, p3, p4, p5, p6, p7,
             ncol=2)


##plotting Research against cgainst Chance
ggplot(data = admissions, aes(x = factor(Research), y = Chance)) + geom_boxplot() + geom_smooth(
  method = "lm", se = "FALSE", color = "black"
)


## Removing serial number column
admissions$Serial.No. = NULL

##
sample_size <- floor(0.75 * nrow(admissions))

train_ind <- sample(seq_len(nrow(admissions)), size = sample_size)

train <- admissions[train_ind, ]
test <- admissions[-train_ind, ]
x.test <- test[1:(length(test)-1)]
y.test <- test$Chance.of.Admit

##testing linear model
lm.model <- lm(Chance.of.Admit~., data = train)
summary(lm.model)
lmmodel.pred <- predict.lm(lm.model, newdata = x.test)
lmmodel_MSE <- mean((lmmodel.pred-y.test)^2)
lmmodel_MSE

##plotting residuals checking assumption
resid <- lm.model$residuals
fnorm <- fitdist(resid, distr = "norm")
train.chance <- train$Chance.of.Admit
plot(train.chance, resid, xlab = "Chance of admission", ylab = "Residuals", main = "Residuals against Chance of admission")
par(mar=c(1,1,1,1))
plot(fnorm)

## Applying best subset selection
regfit <- regsubsets(Chance.of.Admit~., data = train)
sumreg <- summary(regfit)
sumreg

##plotting 
plot(sumreg$bic,  type = 'b', xlab = "No. of predictors", ylab = "BIC", main = "BIC against No. of predictors")
plot(sumreg$cp, type = 'b', xlab = "No. of predictors", ylab = "CP", main = "CP against No. of predictors")
plot(sumreg$adjr2, type = 'b', xlab = "No. of predictors", ylab = "Adjusted R square", main = "Adjusted R square against No. of predictors")
data.frame(
  Adj.R2 = which.max(sumreg$adjr2),
  CP = which.min(sumreg$cp),
  BIC = which.min(sumreg$bic)
)
coef(regfit, which.max(sumreg$adjr2))

#applying linear regression on selected predictors
subset.model <- lm(Chance.of.Admit~ GRE.Score + TOEFL.Score + LOR + CGPA + Research, data = train)
summary(subset.model)

##checking residuals
resid2 <- subset.model$residuals
fnorm2 <- fitdist(resid2, distr = "norm")
plot(train$Chance.of.Admit, resid2, xlab = "Chance of admission", ylab = "Residuals", main = "Residuals against Chance of admission")
par(mar=c(1,1,1,1))
plot(fnorm2)


##predict using linear regression and finding MSE
subset.pred <- predict.lm(subset.model, newdata = test)
subset_MSE <- mean((subset.pred-Chance)^2)
subset_MSE


## Splitting to train and testing set 
x.train <- model.matrix(train$Chance.of.Admit~., data = train)
y.train <- train$Chance.of.Admit
grid <- 10^seq(10, -2, length = 100)

x.test <- model.matrix(test$Chance.of.Admit~., data = test)
y.test <- test$Chance.of.Admit

##RIDGE REGRESSION
ridge.mod <- glmnet(x.train, y.train, alpha = 0, lambda = grid)
cvrr.out <- cv.glmnet(x.train, y.train, alpha = 0)
best_lambda <- cvrr.out$lambda.min
best_lambda

ridge.pred <- predict(ridge.mod, s = best_lambda, newx = x.test)
ridge_MSE <- mean((ridge.pred-y.test)^2)  

out.rr <- glmnet(x.train,y.train,alpha=0) 
predict(out.rr, type="coefficients", s=best_lambda)

ridge_MSE

##LASSO REGRESSION
lasso.mod <- glmnet(x.train, y.train, alpha = 1, lambda = grid)
cvrr.out2 <- cv.glmnet(x.train, y.train, alpha = 1)
best_lambda2 <- cvrr.out2$lambda.min
best_lambda2

lasso.pred <- predict(lasso.mod, s = best_lambda2, newx = x.test)
lasso_MSE <- mean((lasso.pred-y.test)^2)
lasso_MSE
out.rr2 <- glmnet(x.train, y.train, alpha = 1)
predict(out.rr2, type = "coefficients", s = best_lambda2)
