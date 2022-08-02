library(SharedForest)
library(ggplot2)
library(dplyr)
library(BART)
library(gridExtra)
x <- read.csv("CGIAR_model/Data/cgiar_train.csv")
str(x)
all.outcomes <- read.csv("CGIAR_model/Data/target/hours_worked_7_all.csv")
str(all.outcomes)

dim(x)

dim(all.outcomes)

#x and test.outcomes have same number of rows
#assume same observations

merged <- merge(x, all.outcomes, by = "msisdn")

qplot(hours_worked, data = all.outcomes, fill = as.factor(gender), geom = "density", alpha = I(.5))


#replace NA's with 0
#note: this is problematic if a missing value doesn't actually imply a value of 0 for that field
#pick out (manually) columns for which this is appropriate
replace_w_0 <- colnames(merged) #for now, apply to all
merged[,replace_w_0 ][is.na(merged[,replace_w_0 ])] <- 0


#then, remove columns that have too many NA's

merged <- merged[, apply(merged, 2, function(x){sum(is.na(x))})/nrow(merged) < 0.5]

fitmat <- list()
nrep <- 10

for (r in 1:nrep){
train.idx <- sample(1:nrow(merged), .5*nrow(merged))
train.df <- merged[train.idx,]
test.df <- merged[-train.idx,]

#transform x variables to live in [0,1]
transform_x <- function(x){(x - min(x))/(max(x)- min(x))}

train.x <- train.df %>% 
            select(-c(msisdn, gender, hours_worked,SITE_ID, NAME_3, NAME_1,OUT_RATIO_VOICE_COUNT,OUT_RATIO_VOICE_AMOUNT)) %>% #remove non-features
            mutate_all(transform_x) %>%as.matrix
test.x <- test.df %>% 
  select(-c(msisdn, gender, hours_worked,SITE_ID, NAME_3, NAME_1,OUT_RATIO_VOICE_COUNT,OUT_RATIO_VOICE_AMOUNT)) %>% #remove non-features
  mutate_all(transform_x)  %>%as.matrix

#gather outcome values

#standardize continuous value
train.y <- (train.df$hours_worked - mean(train.df$hours_worked))/sd(train.df$hours_worked)
#important: using train.df below is intentional
test.y <- (test.df$hours_worked- mean(train.df$hours_worked))/sd(train.df$hours_worked)

train.delta <- train.df$gender 
test.delta <- test.df$gender


# SHARED FOREST MODEL
opts <- SharedForest::Opts(num_burn = 1000, num_thin = 1, num_save = 1000, num_print = 100)

hypers <- SharedForest::Hypers(X = train.x, 
                               Y = train.y, 
                               W = train.x, 
                               delta = train.delta, 
                               num_tree = 200)

sb  <- SharedBart(X = train.x, #Training covariate matrix for continuous response
                  Y = train.y, #Training continuous response (standardized)
                  W = train.x, #Training covariate matrix for binary response (likely the same as X)
                  delta = train.delta, #Training binary response
                  X_test = test.x, #Testing X
                  W_test = test.x, #Testing X
                  hypers_ = hypers, opts_ = opts)

# Shared forest predictions
delta_star <- (sb$theta_hat_test %>%
                 apply(1:2, function(x){rbinom(n = 1, size = 1, prob = pnorm(x))}))
#Estimate E(Y | delta = 1)
#apply() takes the sum across each mcmc sample r ONLY FOR DELTA_STAR = 1
#mean() takes the mean across posterior samples --> posterior mean
sb_y_samps1 <- apply(delta_star*sb$mu_hat_test, 1, sum) / apply(delta_star, 1, sum)
sb_y_pred1 <- mean(sb_y_samps1)
sb_y_ci1 <- quantile(sb_y_samps1, c(.025, .975))
#Estimate E(Y | delta = 0)
#apply() takes the sum across each mcmc sample r ONLY FOR DELTA_STAR = 0
#mean() takes the mean across posterior samples --> posterior mean
sb_y_samps0 <- apply((1-delta_star)*sb$mu_hat_test, 1, sum) / apply((1-delta_star), 1, sum)
sb_y_pred0 <- meansb_y_samps0
sb_y_ci1 <- quantile(sb_y_samps0, c(.025, .975))

# INDIVIDUAL BART MODELS

# Individual BART model for binary outcome
b1 <- gbart(x.train = train.x, y.train = train.delta, x.test = test.x, type = "pbart", printevery=10000)

# Subsample of testing where delta is predicted to be 0
predicted_theta <- b1$yhat.test %>% apply(2, mean)
b_idx0 <- which(predicted_theta < 0)

# Individual BART models for continuous outcome
b2.1 <- gbart(x.train = train.x[train.delta == 0,],
              y.train = train.y[train.delta == 0],
              x.test = test.x[b_idx0,],
              type = "wbart",
              printevery=10000)
b2.2 <- gbart(x.train = train.x[train.delta == 1,],
              y.train = train.y[train.delta == 1],
              x.test = test.x[-b_idx0,],
              type = "wbart",
              printevery=10000)
# BART predictions
b_delta_pred <- ifelse(pnorm(b1$yhat.test %>% apply(2, mean)) > 0.5, 1, 0)
b_y_pred_delta_0 <- (b2.1$yhat.test %>% apply(2, mean))
b_y_pred_delta_1 <- (b2.2$yhat.test %>% apply(2, mean))




# COMPUTE LOSS FUNCTIONS FOR COMPETING MODELS
# shared forest
l0 <- (mean(test.y[test.delta == 0])-sb_y_pred0)^2
l1 <- (mean(test.y[test.delta == 1])-sb_y_pred1)^2
sf_loss <- mean(c(l0, l1)) #avg of squared error loss

# chained bart
l0 <- (mean(test.y[test.delta == 0])-mean(b_y_pred_delta_0))^2
l1 <- (mean(test.y[test.delta == 1])-mean(b_y_pred_delta_1))^2
b_loss <- mean(c(l0, l1))



sb_y_pred <- sb$mu_hat_test %>% apply(2, mean, na.rm=TRUE)
sb_delta_pred <- ifelse(pnorm(sb$theta_hat_test %>% apply(2, mean)) > 0.5, 1, 0)

fm <- data.frame(rep = r,
                 true_y_delta_0 = mean(test.y[test.delta == 0]),
                 true_y_delta_1 = mean(test.y[test.delta == 1]),
                 b_y_pred_delta_0 = mean(b_y_pred_delta_0),
                 b_y_pred_delta_1 = mean(b_y_pred_delta_1),
                 sb_y_pred_delta_0 = sb_y_pred0,#weighted.mean(sb_y_pred, w = 1-pnorm(sb$theta_hat_test %>% apply(2, mean))),#mean(sb_y_pred[sb_delta_pred == 0]),
                 sb_y_pred_delta_1 = sb_y_pred1,#weighted.mean(sb_y_pred, w = pnorm(sb$theta_hat_test %>% apply(2, mean))),#mean(sb_y_pred[sb_delta_pred == 1]),
                 b_delta_pred_acc = mean(ifelse(predicted_theta > 0, 1, 0) == test.delta),
                 sb_delta_pred_acc = mean(sb_delta_pred == test.delta),
                 b_Y_pred_mse = mean((c(test.y[b_idx0], test.y[-b_idx0])- c(b_y_pred_delta_0, b_y_pred_delta_1))^2),
                 sb_Y_pred_mse = mean((test.y- sb_y_pred)^2),
                 b_loss = b_loss,
                 sf_loss = sf_loss)
#write.csv(ar, paste0("ar_temp", r, "_", overall_reps, ".csv"))


print(fm)
fitmat[[r]] <- fm

}

# Save results
fitmatd <- do.call("rbind", fitmat)

write.csv(fitmatd,"fitmatd_0622.csv", row.names=FALSE)

ar <- data.frame(rep = 1,
                 true_Y = c(test.y[b_idx0], test.y[-b_idx0]),
                 true_delta = c(test.delta[b_idx0], test.delta[-b_idx0]),
                 sb_theta_pred = c((pnorm(sb$theta_hat_test %>% apply(2, mean)))[b_idx0],(pnorm(sb$theta_hat_test %>% apply(2, mean)))[-b_idx0]),
                 b_theta_pred = c(pnorm(b1$yhat.test %>% apply(2, mean))[b_idx0], pnorm(b1$yhat.test %>% apply(2, mean))[-b_idx0]),
                 sb_Y_pred = c(sb_y_pred[b_idx0], sb_y_pred[-b_idx0]),
                 b_Y_pred = c(b_y_pred_delta_0, b_y_pred_delta_1),
                 sb_delta_pred = c(sb_delta_pred[b_idx0],sb_delta_pred[-b_idx0]),
                 b_delta_pred = ifelse(c(predicted_theta[b_idx0], predicted_theta[-b_idx0]) > 0, 1, 0) )
p1 <-  ggplot() +
  #geom_point(aes(x = true_Y, y = sb_Y_pred),size = 3 ,data = ar[!ar$true_delta == ar$sb_delta_pred,])+
  geom_point(aes(x = true_Y, y = sb_Y_pred),alpha = I(.4),data = ar)+
  geom_abline(aes(intercept = 0, slope = 1)) +
  scale_colour_brewer("SF delta pred", palette = "Dark2") +
  labs(x = "True Y", y = "Shared Forest Predicted Y") +
  theme_bw()
p1

p2 <-  ggplot() +
  #geom_point(aes(x = true_Y, y = b_Y_pred),size = 3 ,data = ar[!ar$true_delta == ar$sb_delta_pred,])+
  geom_point(aes(x = true_Y, y = b_Y_pred), data = ar, alpha = I(.4))+
  geom_abline(aes(intercept = 0, slope = 1)) +
  scale_colour_brewer("BART delta pred", palette = "Dark2")+
  labs(x = "True Y", y = "(Chained) BART Predicted Y") +
  theme_bw()#+


p3 <- grid.arrange(p1, p2, ncol  = 2)


ggplot() +
  #geom_point(aes(x = true_Y, y = b_Y_pred),size = 3 ,data = ar[!ar$true_delta == ar$sb_delta_pred,])+
  geom_point(aes(x = sb_Y_pred, y = b_Y_pred,colour = b_delta_pred), data = ar, alpha = I(.4))+
  geom_abline(aes(intercept = 0, slope = 1))

