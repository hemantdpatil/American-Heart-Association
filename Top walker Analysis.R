library(readr)
library(tidyverse)
library(caret)
library(ROCR)
library(xgboost)

dataset<-read.csv("LuminateDataExport_UTDP2_011818.csv",stringsAsFactors = FALSE)
str(dataset)
glimpse(dataset)
summary(dataset)
              
# Remove NA
dataset<-dataset[!(is.na(dataset$City) | dataset$City==""), ]

dataset<-dataset[!(is.na(dataset$Street) | dataset$Street==""), ]

dataset<-dataset[!(is.na(dataset$TAP_DESC) | dataset$TAP_DESC==""), ]

dataset<-dataset[!(is.na(dataset$TAP_LIFED) | dataset$TAP_LIFED==""), ]

dataset<-dataset[!(is.na(dataset$Team_Member_Goal) | dataset$Team_Member_Goal==""),]

dataset<-dataset[!(is.na(dataset$Event_Year) | dataset$Event_Year==""),]

dataset<-dataset[!(is.na(dataset$State) | dataset$State==""),]

dataset<-dataset[!(is.na(dataset$State) | dataset$State==""),]

#Removed Company Goal
dataset$Company_Goal<-NULL

dataset$Event_Year<-gsub("FY ","",dataset$Event_Year)
dataset$Event_Year<-as.numeric(dataset$Event_Year)

dataset$Fundraising_Goal<-gsub("\\$","",dataset$Fundraising_Goal)
dataset$Fundraising_Goal<-gsub(",","",dataset$Fundraising_Goal)
dataset$Fundraising_Goal<-as.numeric(dataset$Fundraising_Goal)

dataset$Team_Average<-gsub("\\$","",dataset$Team_Average)
dataset$Team_Average<-gsub(",","",dataset$Team_Average)
dataset$Team_Average<-as.numeric(dataset$Team_Average)

dataset$Team_Member_Goal<-gsub("\\$","",dataset$Team_Member_Goal)
dataset$Team_Member_Goal<-gsub(",","",dataset$Team_Member_Goal)
dataset$Team_Member_Goal<-as.numeric(dataset$Team_Member_Goal)

dataset$Team_Total_Gifts<-gsub("\\$","",dataset$Team_Total_Gifts)
dataset$Team_Total_Gifts<-gsub(",","",dataset$Team_Total_Gifts)
dataset$Team_Total_Gifts<-as.numeric(dataset$Team_Total_Gifts)

dataset$Participant_Gifts<-gsub("\\$","",dataset$Participant_Gifts)
dataset$Participant_Gifts<-gsub(",","",dataset$Participant_Gifts)
dataset$Participant_Gifts<-as.numeric(dataset$Participant_Gifts)

dataset$Personal_Gift<-gsub("\\$","",dataset$Personal_Gift)
dataset$Personal_Gift<-gsub(",","",dataset$Personal_Gift)
dataset$Personal_Gift<-as.numeric(dataset$Personal_Gift)

dataset$Total_Gifts<-gsub("\\$","",dataset$Total_Gifts)
dataset$Total_Gifts<-gsub(",","",dataset$Total_Gifts)
dataset$Total_Gifts<-as.numeric(dataset$Total_Gifts)


dataset$Top_Walker<-(ifelse(dataset$Total_Gifts >1000, 1, 0))
dataset$Team_Captain<-as.factor(dataset$Team_Captain)


dataset$Event_Date<-as.Date(dataset$Event_Date,format ="%m/%d/%Y" )
dataset$Event_Year<-as.factor(dataset$Event_Year)
dataset$City<-as.factor(dataset$City)
dataset$Street<-as.factor(dataset$Street)
dataset$TAP_LEVEL<-as.factor(dataset$TAP_LEVEL)
dataset$TAP_DESC<-as.factor(dataset$TAP_DESC)
dataset$TAP_LIFED<-as.factor(dataset$TAP_LIFED)

summary(dataset)

glimpse(dataset)
dataset$Registration_Gift<-as.factor(dataset$Registration_Gift)
dataset$State<-as.factor(dataset$State)

#For modelling
data<-dataset[,c('Fundraising_Goal','Team_Count',
                 'Registration_Gift','MEDAGE_CY','Team_Captain',"DIVINDX_CY",
                 'MEDHINC_CY','MEDDI_CY','TAP_LIFED','Top_Walker')]

summary(data)

write.csv(data,file = "data_clean.csv")



#Models
library(caTools)
set.seed(123)
split = sample.split(data$Top_Walker, SplitRatio = 0.75)
training_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)


# Feature Scaling
str(data)
training_set[c(-3,-5,-9,-10)]<-scale(training_set[c(-3,-5,-9,-10)])
test_set[c(-3,-5,-9,-10)]<-scale(test_set[c(-3,-5,-9,-10)])

training_set[c(-3,-4,-5,-7,-11,-12)]<-scale(training_set[c(-3,-4,-5,-7,-11,-12)])
test_set[c(-3,-4,-5,-7,-11,-12)]<-scale(test_set[c(-3,-4,-5,-7,-11,-12)])


# Logistic Regression

log_classifier<-glm(formula = Top_Walker ~ .,
                    family = binomial,
                    data = training_set)
log_classifier

# Predicting the Test set results
y_pred_log = predict(log_classifier, newdata = test_set[-10])
y_pred_log_bin = ifelse(y_pred_log > 0.5, 1, 0)


y_pred_prob_log = predict(log_classifier, newdata = test_set[-10],type = "response")

# Making the Confusion Matrix
cm_log = table(test_set[, 10], y_pred_log_bin)

# Performance analysis
tn_log <- cm_log[1]
tp_log <- cm_log[4]
fp_log <- cm_log[3]
fn_log <- cm_log[2]

accuracy_log <- (tp_log + tn_log) / (tp_log + tn_log + fp_log + fn_log)
misclassification_rate_log <- 1 - accuracy_log
recall_log <- tp_log / (tp_log + fn_log)
precision_log <- tp_log / (tp_log + fp_log)
null_error_rate_log <- tn_log / (tp_log + tn_log + fp_log + fn_log)
Fmeasure_log <- 2 * precision_log * recall_log / (precision_log + recall_log)
tibble(
  accuracy_log,
  misclassification_rate_log,
  recall_log,
  precision_log,
  null_error_rate_log,
  Fmeasure_log
) %>% 
  transpose() 


confusionMatrix(as.numeric(y_pred_log_bin),test_set$Top_Walker,positive = "Yes")
pred_log<-prediction(y_pred_prob_log,test_set$Top_Walker)
perf_log<-performance(pred_log,"tpr","fpr")
dd_log<-data.frame(FP = perf_log@x.values[[1]], TP = perf_log@y.values[[1]])
auc_log<-performance(pred_log, measure = 'auc')@y.values[[1]]




barplot(varImp(log_classifier)$Overall,names.arg =row.names(varImp(log_classifier)) )

imp_log<-varImp(log_classifier)

ggplot(data=imp_log,aes(x=rownames(imp_log),y=imp_log$Overall))+geom_bar(stat = "identity")



##SVM
library(e1071)
svm_classifier = svm(formula = Top_Walker ~ .,
                     data = training_set,
                     type = 'C-classification',
                     kernel = 'radial')
# Predicting the Test set results
y_pred_svm = predict(svm_classifier, newdata = test_set[-10])



y_pred_prob_svm = predict(svm_classifier, newdata = test_set[-10],type = "prob")

# Making the Confusion Matrix
cm_svm = table(test_set[, 10], y_pred_svm)

# Performance analysis
tn_svm <- cm_svm[1]
tp_svm <- cm_svm[4]
fp_svm <- cm_svm[3]
fn_svm <- cm_svm[2]

accuracy_svm <- (tp_svm + tn_svm) / (tp_svm + tn_svm + fp_svm + fn_svm)
misclassification_rate_svm <- 1 - accuracy_svm
recall_svm <- tp_svm / (tp_svm + fn_svm)
precision_svm <- tp_svm / (tp_svm + fp_svm)
null_error_rate_svm <- tn_svm / (tp_svm + tn_svm + fp_svm + fn_svm)
Fmeasure_svm <- 2 * precision_svm * recall_svm / (precision_svm + recall_svm)



tibble(
  accuracy_svm,
  misclassification_rate_svm,
  recall_svm,
  precision_svm,
  null_error_rate_svm,
  Fmeasure_svm
) %>% 
  transpose() 

confusionMatrix(y_pred_svm,test_set$Top_Walker,positive = "Yes")
pred_svm<-prediction(as.numeric(y_pred_prob_svm),test_set$Top_Walker)
perf_svm<-performance(pred_svm,"tpr","fpr")
dd_svm<-data.frame(FP = perf_svm@x.values[[1]], TP = perf_svm@y.values[[1]])
auc_svm<-performance(pred_svm, measure = 'auc')@y.values[[1]]







# Fitting Random Forest Classification to the Training set
library(randomForest)
classifier = randomForest(x = training_set[-10],
                          y = training_set$Top_Walker,
                          importance = TRUE)
a<-importance(classifier)

b<-sort(a[,4],decreasing = TRUE)


as.data.frame(format(b,scientific=FALSE))


# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-10])
y_pred_prob = predict(classifier, newdata = test_set[-10],type = "response")
# Making the Confusion Matrix
cm = table(test_set[, 10], ifelse(y_pred > 0.5, 1, 0))

confusionMatrix(ifelse(y_pred > 0.5, 1, 0),test_set$Top_Walker,positive = "Yes")
pred<-prediction(y_pred_prob,test_set$Top_Walker)
perf<-performance(pred,"tpr","fpr")
dd_RF<-data.frame(FP = perf@x.values[[1]], TP = perf@y.values[[1]])
auc<-performance(pred, measure = 'auc')@y.values[[1]]

# Performance analysis
tn <- cm[1]
tp <- cm[4]
fp <- cm[3]
fn <- cm[2]

accuracy <- (tp + tn) / (tp + tn + fp + fn)
misclassification_rate <- 1 - accuracy
recall <- tp / (tp + fn)
precision <- tp / (tp + fp)
null_error_rate <- tn / (tp + tn + fp + fn)
Fmeasure <- 2 * precision * recall / (precision + recall)



tibble(
  accuracy,
  misclassification_rate,
  recall,
  precision,
  null_error_rate,
  Fmeasure
) %>% 
  transpose() 

plot(classifier)

####Boosting
dmy <- dummyVars(" ~ .", data = data[-10], fullRank=T)
train <- data.frame(predict(dmy, newdata = data))

train_test<-cbind(train,data[10])

library(caTools)
set.seed(1)
split1 = sample.split(train_test$Top_Walker, SplitRatio = 0.70)
training_data = subset(train_test, split1 == TRUE)
testing_data = subset(train_test, split1 == FALSE)
#====================================================================
######################## Preparing for xgboost
dtrain_best = xgb.DMatrix(as.matrix(training_data[,-22]), 
                          label=as.numeric(training_data[,22]))
dtest_best = xgb.DMatrix(as.matrix(testing_data[,-22]))

xgb_param_adult = list(
  nrounds = c(100),
  eta = 0.057,#eta between(0.01-0.2)
  max_depth = 4, #values between(3-10)
  subsample = 0.7,#values between(0.5-1)
  colsample_bytree = 0.7,#values between(0.5-1)
  num_parallel_tree=1,
  objective='binary:logistic',
  min_child_weight = 1,
  booster='gbtree',
  scale_pos_weight=1
)

res = xgb.cv(xgb_param_adult,
             dtrain_best,
             nrounds=700,   # changed
             nfold=3,           # changed
             early_stopping_rounds=50,
             print_every_n = 10,
             verbose= 1)
best<-res$best_iteration

xgb.fit1<-xgboost(data=dtrain_best,nrounds =10,params = list(scale_pos_weight=1,colsample_bytree = 0.7))
preds_xgb <- ifelse(predict(xgb.fit1, newdata=dtest_best) >= 0.5, 1, 0)
confusion_xgb<-confusionMatrix(unlist(testing_data[22]), preds_xgb,positive = '1')
preds_xgb_roc<-predict(xgb.fit1,newdata=dtest_best,type="prob")
pred_xgb<-prediction(preds_xgb_roc,testing_data$Top_Walker)  
performance_xgb<-performance(pred_xgb,"tpr","fpr")
dd_xgb<-data.frame(FP = performance_xgb@x.values[[1]], TP = performance_xgb@y.values[[1]])
auc_xgb<-performance(pred_xgb, measure = 'auc')@y.values[[1]]


roc_log <-geom_line(data =dd_log, aes(x = FP, y = TP, color = 'log'))
roc_svm <-geom_line(data =dd_svm, aes(x = FP, y = TP, color = 'svm'))
roc_RF <-geom_line(data =dd_RF, aes(x = FP, y = TP, color = 'RF'))
roc_xgb_test <-geom_line(data =dd_xgb, aes(x = FP, y = TP, color = 'XgBoost'))


ggplot()+roc_xgb_test+roc_log+roc_svm+roc_RF+
  xlab('False Positive')+
  ylab('True Positive')+ggtitle("ROC")+
  scale_colour_manual(name="Legend",values=c(XgBoost="red",log='blue',svm='cyan',RF='pink'))

#feature importance
imp<-xgb.importance(feature_names = dimnames(dtrain_best)[[2]],model =xgb.fit1 )
head(imp)
xgb.plot.importance(imp[1:8])

#Basic xgboost tree plot
library(DiagrammeR)
bst <- xgboost(data = dtrain_best, max.depth = 2,
               eta = 1, nthread = 2, nround = 2, objective = "binary:logistic")
#basic xgboost tree
xgb.plot.tree(feature_names = colnames(dtrain_best), model =xgb.fit1,render=FALSE)
