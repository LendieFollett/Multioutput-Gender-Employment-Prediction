#for my use: Rcpp::compileAttributes()

#SharedForestBinary is a modification of the SharedForest package
rm(list = ls())
#library(devtools)
#install_github("LendieFollett/Multivariate-Heterogenous-Response-Prediction/SharedForestBinary-master/SharedForestBinary")
#or
#install.packages("/Users/000766412/OneDrive - Drake University/Documents/GitHub/Multivariate-Heterogenous-Response-Prediction/SharedForestBinary-master/SharedForestBinary",
#                 repos = NULL,
#                 type = "source")
library(SharedForestBinary)
library(dplyr)
library(ggplot2)
library(MASS)
library(BART)
library(caret)
library(randomForest)
library(reshape2)

out <- "/Users/000766412/OneDrive - Drake University/Documents/GitHub/Multivariate-Heterogenous-Response-Prediction/"

#"s the variable selection task becomes more difficult, the model which does not share information is far more sensitive to irrelevant predictors than the model which does share"
P = 150
n_train = 500
n_test = 500
rho <- 0.0 #Note: shared bart assumes Z1, Z2 are independent GIVEN the Xs
nrep <- 25
d <- array(NA, dim = c(n_train, 2))
d_test <- array(NA, dim = c(n_test, 2))
Sigma <-rho*(1-diag(2)) + diag (2)


#see Section 4. Simulation Study in Linero paper
sigma_theta1 <- 4
sigma_theta2 <- 4


f_fun1 <- function(W){10*sin(pi*W[,1]*W[,2]) + 20*(W[,3]- 0.5)^2 + 10*W[,4] + 5*W[,5]}
f_fun2 <- function(W){5*sin(pi*W[,1]*W[,2]) + 25*(W[,3]- 0.5)^2 + 5*W[,4] + 10*W[,5]}

opts <- Opts(num_burn = 5000, num_thin = 1, num_save = 5000, num_print = 1000)

fitmat <- list()

oreps <- c(1:4)
results <-  mclapply(oreps, function(overall_reps){
for(r in 1:nrep){
print(paste0("************* Repetition = ", r, " *************"))

W <- matrix(runif(P*n_train), ncol = P)
W_test <- matrix(runif(P*n_test), ncol = P)

means <- c(rep(sigma_theta1, n_train), rep(sigma_theta2, n_train))*(cbind(f_fun1(W), f_fun2(W))/20-0.7) #%>% apply(2, m_mean)
means_test <-  c(rep(sigma_theta1, n_test), rep(sigma_theta2, n_test))*(cbind(f_fun1(W_test), f_fun2(W_test))/20-0.7)#%>% apply(2, m_mean)


for (i in 1:n_train){d[i,] <- mvrnorm(n = 1, mu=means[i,], Sigma = Sigma)}
delta1 <- ifelse(d[,1] > 0,1,0)
delta2 <- ifelse(d[,2] > 0,1,0)

cor(delta1, delta2)

for (i in 1:n_test){d_test[i,] <- mvrnorm(n = 1, mu=means_test[i,], Sigma = Sigma)}
delta1_test <- ifelse(d_test[,1] > 0,1,0)
delta2_test <- ifelse(d_test[,2] > 0,1,0)



hypers <- Hypers(W,delta1, delta2, num_tree = 100)

#BART with shared forest model
sb <- SharedBartBinary(W = W,
           delta1=delta1,
           delta2=delta2,
            W_test=W_test,
           hypers_ = hypers,
           opts_ = opts)


#dirichlet probability posterior means - should pick out most important
#s_hat <- sb$s %>%apply(2, mean)
#ggplot() + geom_bar(aes(x = 1:ncol(W), y = s_hat), stat = "identity") +
#  labs(x = "Variable", y = "Inclusion Probability (s-hat)") + scale_x_continuous(breaks = c(1:ncol(W)))


#two individual BART models
#first, train BART (on training data) to predict gender (delta1) (on the testing data)
b1 <- gbart(x.train = W,
            y.train = delta1,
            x.test = W_test, #get PREDICTED genders for the test set
            type = "pbart",
            printevery=1000)

#subsample of testing where delta1 is predicted to be 0
predicted_theta1 <- b1$yhat.test%>%apply(2, mean)
b_idx0 <- which(predicted_theta1 < 0)


#"use the subsample of users predicted as women <delta1-hat = 0> to retrain the models to classify phone users"
b2.1 <- gbart(x.train = W[delta1 == 0,],
            y.train = delta2[delta1 == 0],
            x.test = W_test[b_idx0,],
            type = "pbart",
            printevery=1000)

#"use the subsample of users predicted as men <delta1-hat = 1>to retrain the models to classify phone users"
b2.2 <- gbart(x.train =W[delta1 == 1,],
            y.train = delta2[delta1 == 1],
            x.test = W_test[-b_idx0,],
            type = "pbart",
            printevery=1000)



#two individual random forest models
rf1 <- randomForest(x = W,
                    y = as.factor(delta1 ))
predicted_class <- predict(object = rf1, newdata = W_test, type = "class")
rf_idx0 <- which(predicted_class == "0")

#train on delta1 == 0 sample
rf2.1 <- randomForest(x = W[delta1 == 0,],
                    y = as.factor(delta2)[delta1 == 0])
#train on delta1 == 1 sample
rf2.2 <- randomForest(x = W[delta1 == 1,],
                      y = as.factor(delta2)[delta1 == 1])


#Random forest estimated P(d2 = 1 | d1)
rf_d1_pred <- ifelse(predict(object = rf1, newdata = W_test, type = "class") == "1", 1, 0)
rf_d2_pred_d1_0 <- ifelse(predict(object = rf2.1, newdata = W_test[rf_idx0,], type = "class") == "1", 1, 0)
rf_d2_pred_d1_1 <- ifelse(predict(object = rf2.2, newdata = W_test[-rf_idx0,], type = "class") == "1", 1, 0)


#bart estimated  P(d2 = 1 | d1)
b_d1_pred <- ifelse(pnorm(b1$yhat.test%>%apply(2, mean))> 0.5, 1, 0)
b_d2_pred_d1_0 <- ifelse(pnorm(b2.1$yhat.test%>%apply(2, mean))> 0.5, 1, 0)
b_d2_pred_d1_1 <- ifelse(pnorm(b2.2$yhat.test%>%apply(2, mean))> 0.5, 1, 0)

# Shared BART predictions
delta_star <- (sb$theta_hat_test1 %>%
                 apply(1:2, function(x){rbinom(n = 1, size = 1, prob = pnorm(x))}))
#estimated P(delta2 = 1 | delta1 = 1)
sb_y_pred1 <- mean(apply(delta_star*sb$theta_hat_test2, 1, sum) / apply(delta_star, 1, sum)) %>% pnorm()
#estimated P(delta2 = 1 | delta1 = 0)
sb_y_pred0 <- mean(apply((1-delta_star)*sb$theta_hat_test2, 1, sum) / apply((1-delta_star), 1, sum)) %>% pnorm()


#True/observed  P(d2 = 1 | d1) in test set
fm<- data.frame(true_d2_d1_0 = mean(delta2_test[delta1_test == 0]),
                true_d2_d1_1 = mean(delta2_test[delta1_test == 1]),
                rf_d2_pred_d1_0 = mean(rf_d2_pred_d1_0),
                rf_d2_pred_d1_1 = mean(rf_d2_pred_d1_1),
                b_d2_pred_d1_0 = mean(b_d2_pred_d1_0),
                b_d2_pred_d1_1 = mean(b_d2_pred_d1_1),
                sb_d2_pred_d1_0 = sb_y_pred0,
                sb_d2_pred_d1_1 = sb_y_pred1)

fitmat[[r]] <- fm

}
  return(fitmat)
}, mc.cores = length(oreps))


# Store results
fitmatd_1 <- unlist(results, recursive = FALSE)
fitmatd <- do.call("rbind", fitmatd_1)


write.csv(fitmatd, paste0(out, "/binary_sims.csv"), row.names=FALSE)

# Read in results
fitmatd <- read.csv(paste0(out, "/binary_sims.csv"))
fitmatd_long0 <- fitmatd[,c(1,3,5,7)] %>% melt(id.vars = c(1)) %>%
  mutate(variable = factor(variable, levels = c("rf_d2_pred_d1_0","b_d2_pred_d1_0","sb_d2_pred_d1_0"),
                           labels = c("Random Forest", "BART", "Shared Forest")))

fitmatd_long1 <- fitmatd[,c(2,4,6,8)] %>% melt(id.vars = c(1))%>%
  mutate(variable = factor(variable, levels = c("rf_d2_pred_d1_1","b_d2_pred_d1_1","sb_d2_pred_d1_1"),
                           labels = c("Random Forest", "BART", "Shared Forest")))

fitmatd_long0 %>% group_by(variable) %>% summarise(mse = mean((true_d2_d1_0 - value)^2))
fitmatd_long1 %>% group_by(variable) %>% summarise(mse = mean((true_d2_d1_1 - value)^2))

p0 <- fitmatd_long0 %>% ggplot() +
  geom_density(aes(x = (value - true_d2_d1_0)^2, fill = variable, linetype = variable), alpha = I(.3)) +
  labs(x = "Squared Error") +
  scale_linetype("Model")+
  scale_fill_manual("Model",values=c("grey10","grey40", "grey90")) + theme_bw() +
  ggtitle("delta[1] = 0")

p1 <- fitmatd_long1 %>% ggplot() +
  geom_density(aes(x = (value - true_d2_d1_1)^2, fill = variable, linetype = variable), alpha = I(.3))+
  labs(x = "Squared Error")+
  scale_linetype("Model")+
  scale_fill_manual("Model",values=c("grey10","grey40", "grey90")) + theme_bw()+
  ggtitle("delta[1] = 1")
p <- grid.arrange(p0, p1, nrow = 2)
ggsave(paste0(out, "/binary_sim_results.pdf"), plot = p)



