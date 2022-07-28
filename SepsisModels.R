#### loading prepared data:

#load("data_train_2022.rda")
#load("data_test_2022.rda")
#load("sepsis_data_2022.rda")

#### loading necessary libraries:

library(missForest)
library(randomForest)
library(ranger)
library(rsample)
library(mlr)
library(DALEX)
library(xai2shiny)

data_split <- initial_split(sepsis, strata = "surv_28_d", p = 0.8)

data_train <- training(data_split)
data_test  <- testing(data_split)

#### 1 step: Prepare three types of models and compare them:

## Logistic Regression:

logreg <- makeLearner("classif.logreg", predict.type = "prob")

## Random Forest:

randomForest <- makeLearner("classif.randomForest", predict.type = "prob")

## XGBoosting: 

xgboost<- makeLearner("classif.xgboost", predict.type = "prob",
                      par.vals = list(
                        objective = "binary:logistic",
                        eval_metric = "error",
                        nrounds = 200))

# Benchmark - compare performacne (AUC and mmce) of three types of models:

task <- makeClassifTask(data = sepsis_data, target = "surv_28_d") 

perf <- benchmark(
  list(logreg, rf.lrn, xgboost),
  list(task),
  resamplings = makeResampleDesc(method = "RepCV", reps = 5, folds = 5),
  measures = list(mlr::auc, mmce)
)

plotBMRBoxplots(perf)

#### 2 step: Tune parameters for Random Forest model:

set.seed(14)

task <- makeClassifTask(data = sepsis_data,target = "surv_28_d") 
traintask <- makeClassifTask(data = data_train, target = "surv_28_d") 
testtask <- makeClassifTask(data = data_test, target = "surv_28_d")

rf.lrn <- makeLearner("classif.randomForest", predict.type = "prob")
rf.lrn$par.vals <- list( num.trees=500)

params <- makeParamSet(makeIntegerParam("mtry",lower = 2,upper = 5), 
                       makeIntegerParam("nodesize", lower = 1, upper = 10000), 
                       makeIntegerParam("ntree", lower = 100, upper = 10000))

rdesc <- makeResampleDesc("CV",iters = 10)

ctrl <- makeTuneControlRandom(maxit = 100)

tune <- tuneParams(learner = rf.lrn, task = task, resampling = rdesc, measures = mlr::auc, par.set = params, control = ctrl, show.info = T)

rf.lrn$par.vals <- list(ntree=tune$x$ntree, mtry=tune$x$mtry, nodesize=tune$x$nodesize)

# model RF:

RF_mlr <- train(rf.lrn, traintask)

#p_mlr <- data.frame(predict(RF_mlr, traintask))[,4]
#pr_mlr <- ROCR::prediction(p_mlr, data_train$surv_28_d)
#auc_mlr <- ROCR::performance(pr_mlr, measure = "auc")
#auc_mlr <- auc_mlr@y.values[[1]]
#auc_mlr

perf_rf <- benchmark(
  list(rf.lrn),
  list(traintask),
  resamplings = makeResampleDesc(method = "RepCV", reps = 10, folds = 10),
  measures = list(mlr::auc, mmce)
)

plotBMRBoxplots(perf_rf)

#### 3 step: Explain Random Forest Model:

preds <- colnames(sepsis_data)[-1]

explainer_rf_variable_test <- DALEX::explain(getLearnerModel(RF_mlr), data = data.frame(data_test[,preds]), y=as.numeric(as.character(data_test$"surv_28_d")),label = "Random Forest test")
explainer_rf_variable_train <- DALEX::explain(getLearnerModel(RF_mlr), data = data.frame(data_train[,preds]), y=as.numeric(as.character(data_train$"surv_28_d")),label = "Random Forest train")
explainer_rf_variable <- DALEX::explain(getLearnerModel(RF_mlr), data = data.frame(sepsis_data[,preds]), y=as.numeric(as.character(sepsis_data$"surv_28_d")), label = "Random Forest")

#### Create plots 

# Feature Importance:

rf_vi <- model_parts(explainer_rf_variable)

plot(rf_vi, show_boxplots = FALSE) + labs( subtitle = "") +
  theme(axis.text=element_text(size=14),
        axis.title=element_text(size=14))+
  theme(plot.title = element_text(size=20))+
  theme(strip.text.x = element_blank())

# ROC curve:

plot(model_performance(explainer_rf_variable), geom = "roc") +
  labs( subtitle = "") +
  theme(legend.position = "none") +
  theme(axis.text=element_text(size=14),
        axis.title=element_text(size=14))+
  theme(plot.title = element_text(size=20))+
  annotate("text", x= 0.75, y = 0.2, label = "AUC = 0.92", color = "#4378bf", size = 5) +
  theme(plot.margin = unit(c(0,0,0,0), "cm"))

# PDP:

rf_mprofile <- model_profile(explainer = explainer_rf_variable, variables = c("pFN"), type = "partial")

plot(rf_mprofile) +
  theme(axis.text=element_text(size=14),
        axis.title=element_text(size=14))+
  theme(plot.title = element_text(size=20)) +
  labs( subtitle = "pFN") +
  theme(strip.text.x = element_blank())

# Local Expaliners:

#choose the observation:

new_observation_21_plt <- sepsis_data[20,]

# Break Down:

plot(predict_parts(explainer_rf_variable, new_observation = new_observation_21_plt)) +
  ggtitle("Break down effects") + 
  labs( subtitle = "survivor")+
  theme(strip.text.x = element_blank(), strip.text.y = element_blank(), strip.text = element_blank())  +
  theme(axis.text=element_text(size=14),
        axis.title=element_text(size=14))+
  theme(plot.title = element_text(size=20))

# SHAP:

plot(predict_parts(explainer_rf_variable, new_observation = new_observation_21_plt,type= "shap")) +
  ggtitle("SHAP values") + 
  labs( subtitle = "survivor")+
  theme(strip.text.x = element_blank(), strip.text.y = element_blank(), strip.text = element_blank())  +
  theme(axis.text=element_text(size=14),
        axis.title=element_text(size=14))+
  theme(plot.title = element_text(size=20))

# Ceteris Paribus:

plot(predict_profile(explainer_rf_variable, new_observation = new_observation_21_plt)) +
  ggtitle("Ceteris Paribus") + 
  labs( subtitle = "survivor")+
  #theme(strip.text.x = element_blank(), strip.text.y = element_blank(), strip.text = element_blank()) +
  theme(axis.text=element_text(size=14),
        axis.title=element_text(size=14))+
  theme(plot.title = element_text(size=20))

ggsave("Figure6_2022.pdf", width = 8, height = 5, device='pdf', dpi=300)

new_observation_57_plt <- sepsis_data[57,]

plot(predict_parts(explainer_rf_variable, new_observation = new_observation_57_plt)) +
  ggtitle("Break down effects") + 
  labs( subtitle = "nonsurvivor")+
  theme(strip.text.x = element_blank(), strip.text.y = element_blank(), strip.text = element_blank())  +
  theme(axis.text=element_text(size=14),
        axis.title=element_text(size=14))+
  theme(plot.title = element_text(size=20))
ggsave("Figure7_2022.pdf", width = 8, height = 5, device='pdf', dpi=300)

plot(predict_parts(explainer_rf_variable, new_observation = new_observation_57_plt, type= "shap")) +
  ggtitle("Break down effects") + 
  labs( subtitle = "nonsurvivor")+
  theme(strip.text.x = element_blank(), strip.text.y = element_blank(), strip.text = element_blank())  +
  theme(axis.text=element_text(size=14),
        axis.title=element_text(size=14))+
  theme(plot.title = element_text(size=20))

plot(predict_profile(explainer_rf_variable, new_observation = new_observation_57_plt))  +
  ggtitle("Ceteris Paribus") + 
  labs( subtitle = "nonsurvivor")+
  #theme(strip.text.x = element_blank(), strip.text.y = element_blank(), strip.text = element_blank()) +
  theme(axis.text=element_text(size=14),
        axis.title=element_text(size=14))+
  theme(plot.title = element_text(size=20))


#### Create basic application 

# devtools::install_github("ModelOriented/xai2shiny")

library(xai2shiny)

xai2shiny(explainer_rf_variable, explainer_rf_variable_test)
xai2shiny(explainer_rf_variable, explainer_rf_variable_test, directory="/Users/kasia/Projekty/SEPSA/Shiny1/")
xai2shiny(explainer_rf_variable, explainer_logr, explainer_rf_variable_test)

