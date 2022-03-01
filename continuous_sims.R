# Import packages
rm(list = ls())
#remove.packages("SharedForest")
#library(devtools)
#install_github("theodds/SharedForestPaper/SharedForest")
library(SharedForest)
library(dplyr)
library(ggplot2)
library(MASS)
library(BART)
library(caret)
library(randomForest)
library(reshape2)
library(parallel)
library(gridExtra)
# Set file path
#out = "/Users/hendersonhl/Documents/Articles/Multivariate-Heterogenous-Response-Prediction"
out = "/Users/000766412/OneDrive - Drake University/Documents/GitHub/Multivariate-Heterogenous-Response-Prediction"
# Set parameters


P = 150
n_train = 500
n_test = 500
rho <- 0.0

nrep <- 25
d <- array(NA, dim = c(n_train, 2))
d_test <- array(NA, dim = c(n_test, 2))
Sigma <-rho*(1-diag(2)) + diag (2)

sigma_theta <- 4

# Define helper functions

f_fun1 <- function(W){10*sin(pi*W[,1]*W[,2]) + 20*(W[,3]- 0.5)^2 + 10*W[,4] + 5*W[,5]}
f_fun2 <- function(W){5*sin(pi*W[,1]*W[,2]) + 25*(W[,3]- 0.5)^2 + 5*W[,4] + 10*W[,5]}

# Set up for simulations
opts <- Opts(num_burn = 5000, num_thin = 1, num_save = 5000, num_print = 1000)
fitmat <- list()
allresults <- list()

oreps <- c(1:4)
results <-  mclapply(oreps, function(overall_reps){
  # Start simulations
  for(r in 1:nrep){
    print(paste0("************* Repetition = ", r, " *************"))
    start.time <- Sys.time()
    
    # Generate covariates
    W <- matrix(runif(P*n_train), ncol = P)
    W_test <- matrix(runif(P*n_test), ncol = P)
    
    # Generate expected values
    means <-       cbind(f_fun1(W),      sigma_theta*(f_fun2(W)/20 - 0.7))
    means_test <-  cbind(f_fun1(W_test), sigma_theta*(f_fun2(W_test)/20 - 0.7))
    
    # Generate training outcomes
    for (i in 1:n_train){d[i,] <- mvrnorm(n = 1, mu = means[i,], Sigma = Sigma)}
    Y <- (d[,1] - mean(d[,1]))/ sd(d[,1])
    delta <- ifelse(d[,2] > 0, 1, 0)
    
    # Generate testing outcomes
    for (i in 1:n_test){d_test[i,] <- mvrnorm(n = 1, mu = means_test[i,], Sigma = Sigma)}
    # Standardize using training summary statistics
    # (Predicting how many training sd's above or below training mean test set obs falls)
    Y_test <- (d_test[,1] - mean(d[,1]))/ sd(d[,1])
    delta_test <- ifelse(d_test[,2] > 0, 1, 0)
    
    # Shared forest model
    hypers <- SharedForest::Hypers(X = W, Y = Y, W = W, delta = delta, num_tree = 200)
    
    sb  <- SharedBart(X = W, Y = Y, W = W, delta = delta, X_test = W_test,
                      W_test = W_test, hypers_ = hypers, opts_ = opts)
    
    #s_hat <- sb$s %>%apply(2, mean)
    #ggplot() + geom_bar(aes(x = 1:ncol(W), y = s_hat), stat = "identity") +
    #  labs(x = "Variable", y = "Inclusion Probability (s-hat)") + scale_x_continuous(breaks = c(1:ncol(W)))
    
    # Individual BART model for binary outcome
    b1 <- gbart(x.train = W, y.train = delta, x.test = W_test, type = "pbart", printevery=10000)
    
    # Subsample of testing where delta is predicted to be 0
    predicted_theta <- b1$yhat.test %>% apply(2, mean)
    b_idx0 <- which(predicted_theta < 0)
    
    # Individual BART models for continuous outcome
    b2.1 <- gbart(x.train = W[delta == 0,],
                  y.train = Y[delta == 0],
                  x.test = W_test[b_idx0,],
                  type = "wbart",
                  printevery=10000)
    b2.2 <- gbart(x.train = W[delta == 1,],
                  y.train = Y[delta == 1],
                  x.test = W_test[-b_idx0,],
                  type = "wbart",
                  printevery=10000)
    
    #LRF: for every mcmc sample, generate delta*_i i = 1, ..., N_test.
    
    # Shared BART predictions
    delta_star <- (sb$theta_hat_test %>%
                     apply(1:2, function(x){rbinom(n = 1, size = 1, prob = pnorm(x))}))
    sb_y_pred1 <- mean(apply(delta_star*sb$mu_hat_test, 1, sum) / apply(delta_star, 1, sum))
    sb_y_pred0 <- mean(apply((1-delta_star)*sb$mu_hat_test, 1, sum) / apply((1-delta_star), 1, sum))
    
    sb_y_pred <- sb$mu_hat_test %>% apply(2, mean, na.rm=TRUE)
    sb_delta_pred <- ifelse(pnorm(sb$theta_hat_test %>% apply(2, mean)) > 0.5, 1, 0)
    
    # BART predictions
    b_delta_pred <- ifelse(pnorm(b1$yhat.test %>% apply(2, mean)) > 0.5, 1, 0)
    b_y_pred_delta_0 <- b2.1$yhat.test %>% apply(2, mean)
    b_y_pred_delta_1 <- b2.2$yhat.test %>% apply(2, mean)
    
    ar <- data.frame(rep = r,
                     true_Y = c(Y_test[b_idx0], Y_test[-b_idx0]),
                     true_delta = c(delta_test[b_idx0], delta_test[-b_idx0]),
                     sb_theta_pred = c((pnorm(sb$theta_hat_test %>% apply(2, mean)))[b_idx0],(pnorm(sb$theta_hat_test %>% apply(2, mean)))[-b_idx0]),
                     b_theta_pred = c(pnorm(b1$yhat.test %>% apply(2, mean))[b_idx0], pnorm(b1$yhat.test %>% apply(2, mean))[-b_idx0]),
                     sb_Y_pred = c(sb_y_pred[b_idx0], sb_y_pred[-b_idx0]),
                     b_Y_pred = c(b_y_pred_delta_0, b_y_pred_delta_1),
                     sb_delta_pred = c(sb_delta_pred[b_idx0],sb_delta_pred[-b_idx0]),
                     b_delta_pred = ifelse(c(predicted_theta[b_idx0], predicted_theta[-b_idx0]) > 0, 1, 0) )
    
    #ar %>% group_by(true_delta) %>% summarise(sb = mean(!true_delta == sb_delta_pred),b = mean(!true_delta == b_delta_pred))
    
    # True and observed  P(y | delta) in test set
    fm <- data.frame(rep = r,
                     true_y_delta_0 = mean(Y_test[delta_test == 0]),
                     true_y_delta_1 = mean(Y_test[delta_test == 1]),
                     b_y_pred_delta_0 = mean(b_y_pred_delta_0),
                     b_y_pred_delta_1 = mean(b_y_pred_delta_1),
                     sb_y_pred_delta_0 = sb_y_pred0,#weighted.mean(sb_y_pred, w = 1-pnorm(sb$theta_hat_test %>% apply(2, mean))),#mean(sb_y_pred[sb_delta_pred == 0]),
                     sb_y_pred_delta_1 = sb_y_pred1,#weighted.mean(sb_y_pred, w = pnorm(sb$theta_hat_test %>% apply(2, mean))),#mean(sb_y_pred[sb_delta_pred == 1]),
                     b_delta_pred_acc = mean(ifelse(predicted_theta > 0, 1, 0) == delta_test),
                     sb_delta_pred_acc = mean(sb_delta_pred == delta_test),
                     b_Y_pred_mse = mean((c(Y_test[b_idx0], Y_test[-b_idx0])- c(b_y_pred_delta_0, b_y_pred_delta_1))^2),
                     sb_Y_pred_mse = mean((Y_test- sb_y_pred)^2))
    #write.csv(ar, paste0("ar_temp", r, "_", overall_reps, ".csv"))
    
    
    print(fm)
    fitmat[[r]] <- fm
    allresults[[r]] <- ar
    end.time <- Sys.time()
    time.taken <- end.time - start.time
    print(time.taken)
  }
  
  return(fitmat)
}, mc.cores = length(oreps))


# Save results
fitmatd_1 <- unlist(results, recursive = FALSE)
fitmatd <- do.call("rbind", fitmatd_1)

path <- paste0(out, "/continuous_sims.csv")
write.csv(fitmatd, path, row.names = FALSE)



# Import results and organize for plotting
path <- paste0(out, "/continuous_sims.csv")
fitmatd <- read.csv(path)

# Visualize results

fitmatd_long0 <- fitmatd[,c(2,4,6)] %>% melt(id.vars = c(1)) %>%
  mutate(variable = factor(variable, levels = c("b_y_pred_delta_0","sb_y_pred_delta_0"),
                           labels = c( "BART", "Shared Forest"))) %>%
  rename(true_y = true_y_delta_0)

fitmatd_long1 <- fitmatd[,c(3,5,7)] %>% melt(id.vars = c(1))%>%
  mutate(variable = factor(variable, levels = c("b_y_pred_delta_1","sb_y_pred_delta_1"),
                           labels = c( "BART", "Shared Forest")))%>%
  rename(true_y = true_y_delta_1)

p <- rbind(data.frame(fitmatd_long0, delta = 0),
           data.frame(fitmatd_long1, delta = 1)) %>%
  ggplot() +
  geom_density(aes(x = (value - true_y)^2, fill = variable, linetype = variable), alpha = I(.3)) +
  facet_wrap(~delta, ncol = 1) +
  labs(x = "Squared Error") +
  scale_linetype("Model")+
  scale_fill_manual("Model",values=c("grey10","grey40", "grey90")) + theme_bw()
p

ggsave(paste0(out, "/continuous_sim_results.pdf"), plot = p)

rbind(data.frame(fitmatd_long0, delta = 0),
      data.frame(fitmatd_long1, delta = 1)) %>%
  group_by(variable, delta)%>%
  summarise(MSE = mean((true_y - value)^2),
            L_bound = quantile((true_y - value)^2, .025),
            U_bound = quantile((true_y - value)^2, 0.975))


rbind(data.frame(fitmatd_long0, delta = 0),
      data.frame(fitmatd_long1, delta = 1)) %>%
  ggplot() +geom_point(aes(x = true_y, y = value, colour = variable)) +
  geom_abline(aes(intercept = 0, slope = 1))


# Predicting individual-level responses

fitmatd[,c(1,10,11)] %>% melt(id.vars = c(1)) %>%
  mutate(variable = factor(variable, levels = c("b_Y_pred_mse","sb_Y_pred_mse"),
                           labels = c( "BART", "Shared Forest"))) %>%
  ggplot() + 
  geom_density(aes(x = value, fill = variable))


fitmatd[,c(1,8,9)] %>% melt(id.vars = c(1)) %>%
  mutate(variable = factor(variable, levels = c("b_delta_pred_acc","sb_delta_pred_acc"),
                           labels = c( "BART", "Shared Forest"))) %>%
  ggplot() + 
  geom_density(aes(x = value, fill = variable))

############################################
p1 <-  ggplot() +
  geom_point(aes(x = true_Y, y = sb_Y_pred),size = 3 ,data = ar[!ar$true_delta == ar$sb_delta_pred,])+
  geom_point(aes(x = true_Y, y = sb_Y_pred,  colour = as.factor(sb_delta_pred)),data = ar)+
  geom_abline(aes(intercept = 0, slope = 1)) +
  geom_vline(aes(xintercept = c(mean(Y_test[delta_test == 0]),mean(Y_test[delta_test == 1])))) +
  geom_hline(aes(yintercept = weighted.mean(sb_y_pred, w = 1-pnorm(sb$theta_hat_test %>% apply(2, mean))))) +
  geom_hline(aes(yintercept = weighted.mean(sb_y_pred, w = pnorm(sb$theta_hat_test %>% apply(2, mean))))) +
  scale_colour_brewer("SB delta pred", palette = "Dark2") +
  geom_text(aes(x = 2, y = 2, label = paste0("Error = ", mean(!ar$true_delta == ar$sb_delta_pred))))
p1

p2 <-  ggplot() +
  geom_point(aes(x = true_Y, y = b_Y_pred),size = 3 ,data = ar[!ar$true_delta == ar$sb_delta_pred,])+
  geom_point(aes(x = true_Y, y = b_Y_pred, colour = as.factor(b_delta_pred)), data = ar)+
  geom_abline(aes(intercept = 0, slope = 1)) +
  geom_vline(aes(xintercept = c(mean(Y_test[delta_test == 0]),mean(Y_test[delta_test == 1])))) +
  geom_hline(aes(yintercept = mean(b_y_pred_delta_0))) +
  geom_hline(aes(yintercept = mean(b_y_pred_delta_1)))+
  scale_colour_brewer("CB delta pred", palette = "Dark2")+
  geom_text(aes(x = 2, y = 2, label = paste0("Error = ", mean(!ar$true_delta == ar$b_delta_pred))))

p3 <- grid.arrange(p1, p2)
#ggsave("example_result.pdf", plot = p3, width = 8, height = 10)


ggplot(data = ar[!ar$true_delta == ar$sb_delta_pred,]) +
  geom_density(aes(x = true_Y, fill = as.factor(true_delta)),alpha = I(.4)) +
  xlim(-2.5, 2.5)+
  scale_fill_brewer("TRUE delta", palette = "Dark2")

ggplot(data = ar[!ar$true_delta == ar$b_delta_pred,]) +
  geom_density(aes(x = true_Y, fill = as.factor(true_delta)),alpha = I(.4))+
  xlim(-2.5, 2.5)+
  scale_fill_brewer("TRUE delta ", palette = "Dark2")


