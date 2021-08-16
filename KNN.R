# Progress Report 1
# Predicting Future Sales
# KNN Method 

library(dplyr)
library(lubridate)
library(class)
library(tsfknn)
library(ggplot2)
library(pROC)
library(mlbench)
library(caret)

# load data 
sales <- read.csv(file.choose(), sep=",", header=TRUE)
items <- read.csv(file.choose(), sep=",", header=TRUE)
str(sales)
str(items)

# join the two datasets to have category id in sales df
sales <- merge(sales, items[, c("item_id", "item_category_id")],
               by ="item_id", all.x = TRUE)
# changing the format to date instead of character so we can take more info
sales$date <- as.Date(sales$date, "%d.%m.%Y")
str(sales)
sales$shop_id <- as.factor(sales$shop_id)

# create some features from date info
sales$year <- as.factor(year(sales$date))
sales$month <- as.factor(month(sales$date))
sales$week <- as.factor(week(sales$date))
sales$day <- as.factor(day(sales$date))
sales$weekdays <- as.factor(weekdays(sales$date))
sales$is_weekend <- as.factor(ifelse(sales$weekdays == "Saturday" | 
                                       sales$weekdays == "Sunday", 1, 0))

# regroup data by shop to get an idea of where sales are coming from 
shop.sales <- as.data.frame( sales %>%
                               select(shop_id, item_cnt_day) %>%
                               group_by(shop_id) %>%
                               summarise(item_cnt_day =  sum(item_cnt_day, na.rm = TRUE)) )

shop.sales
ggplot(data=shop.sales, aes(shop_id, item_cnt_day)) + geom_col() + xlab("Shop ID") + ylab("Overall Item Sales")

# regroup by item 
items.sales <- as.data.frame(sales %>%
                               select(item_category_id, item_cnt_day) %>%
                               group_by(item_category_id) %>%
                               summarise(item_cnt_day =  sum(item_cnt_day, na.rm = TRUE)) )
items.sales

# look at weekly sales
weekly.sales <- as.data.frame(sales %>%
                                group_by(year, week) %>%
                                summarise(total_sales =  sum(item_cnt_day, na.rm = TRUE)))

# look at daily sales 
daily.sales <-  as.data.frame(sales %>%
                                group_by(year, month, day) %>%
                                summarise(total_sales =  sum(item_cnt_day, na.rm = TRUE)))
head(daily.sales)
# create time series from daily sales
z.daily.ts <- ts(daily.sales[,4],
               frequency=365,
               start(2013, 1))
plot(z.daily.ts, ylab="Item Sales", xlab="Time", xlim=c(2, 5))


# will first try to predict october 2015 TOTAL sales (not by shop)
monthly.sales <- as.data.frame(sales %>%
                                 group_by(year, month) %>%
                                 summarise(total_sales =  sum(item_cnt_day, na.rm = TRUE)))
plot(monthly.sales$total_sales)


###########################################################
# determining which method to use

t.monthly.sales <- monthly.sales[1:33,] # leaving out october to test accuracy
actual.october <- monthly.sales[34,3]
month.ts <- ts(t.monthly.sales[,3], # creating a time series
               frequency=12, # for every month 
               start=c(2013,1)) # starting in Jan 2013

# building model 
knn.month <- knn_forecasting(month.ts, h=1, lags=1:12, k=3, cf="mean")
knn_examples(knn.month)
# saving prediction 
pred.m.october <- knn.month$prediction[1]
plot(knn.month)
# plot to see what neighbors are used
autoplot(knn.month, highligh="neighbors", faceting=FALSE)

# calculate error and save
error.month <- actual.october - pred.m.october
error.month

##########################################################
#daily approach
t.daily.sales <- daily.sales[1:1003,] #leaving out october to test accuracy

# create a time series
daily.ts <- ts(t.daily.sales[,4],
               frequency=365,
               start(2013, 1))
plot(daily.ts)

# generate model to predict next 30 days
knn.daily <- knn_forecasting(daily.ts, h=30, lags=1:365, k=30, msas="MIMO", cf="mean")
plot(knn.daily)
# plot prediction
autoplot(knn.daily, faceting=FALSE)
# save prediction
pred.d.october <- sum(knn.daily$prediction)
# calculate and save error
error.daily <- actual.october - pred.d.october
error.daily
###############################################################################
# daily arima (recursive)
knn.daily.arima <- knn_forecasting(daily.ts, h=30, lags=1:365, k=30,
                                   msas="recursive", cf="mean")
# plot prediction
autoplot(knn.daily.arima, faceting=FALSE)
# save prediction
pred.da.october <- sum(knn.daily.arima$prediction)
# calcualte and save error
error.daily.a <- actual.october - pred.da.october
error.daily.a
# compare errors
error.month
error.daily
error.daily.a
#################################################################################
# for loop to determine what k value to use for monthly model
for (i in 0:3) {
  knn.tuned <- knn_forecasting(month.ts, h=1, lags=1:12, k=2*i+1)
  pred1 <- knn.tuned$prediction[1]
  err1 <- actual.october - pred1
  print(paste0("For K =", 2*i+1, " the error is ", err1))
}

# given that k=3 is optimum, use same model as before (same code as before here)
knn.month <- knn_forecasting(month.ts, h=1, lags=1:12, k=3)
knn_examples(knn.month)
pred.m.october <- knn.month$prediction[1]
plot(knn.month)
autoplot(knn.month, highligh="neighbors", faceting=FALSE)

error.month <- actual.october - pred.m.october
error.month

###############################################################################
# generate grid to look at different parameters for daily model
daily.grid <- expand.grid(
  k=c(20, 25, 30, 35, 40, 45),
  msas=c("MIMO", "recursive"),
  cf=c("mean", "median", "weighted"),
  error = 0,
  abs.error = 0
)
nrow(daily.grid)
daily.grid
# changing from numeric to factor
daily.grid[,c(2,3)] <- lapply(daily.grid[,c(2,3)], as.character)
class(daily.grid$cf)

# for loop to make models and iterate through all combinations from grid
for (i in 1:nrow(daily.grid)) {
  daily.tuned <- knn_forecasting(daily.ts, h=30, lags=1:365,
                                 k=daily.grid$k[i],
                                 msas=daily.grid$msas[i],
                                 cf=daily.grid$cf[i])
  pred1.oct <- sum(daily.tuned$prediction)
  daily.grid$error[i] <- actual.october - pred1.oct
  daily.grid$abs.error[i] <- abs(actual.october - pred1.oct)
}

# sort set of parameters by lowest to highest absolute error
daily.grid %>% 
  arrange(abs.error)

#using k=35, recursive, median is best

# daily arima with findings from grid searcg
knn.d.tuned <- knn_forecasting(daily.ts, h=30, lags=1:365, k=35, 
                               msas="recursive", cf="median")
autoplot(knn.d.tuned, faceting=FALSE)
# save preds
pred.da.october <- sum(knn.d.tuned$prediction)
# calculate and save error
error.daily.a <- actual.october - pred.da.october
error.daily.a
# compare errors
error.month
error.daily
error.daily.a

################################################################################
# make models by shop 
# first need to reorganize data 
daily.sales1 <-  as.data.frame(sales %>%
                                select(shop_id, year, month, day, item_cnt_day) %>%
                                group_by(shop_id, year, month, day) %>%
                                summarise(total_sales =  sum(item_cnt_day, na.rm = TRUE)))

sales1 <- daily.sales1

sales1 <- sales1 %>%
  arrange(shop_id, year, month, day)

# making sure we have all the dates for a complete time series
new.daily <- daily.sales[,1:3]
new.daily$total_sales <- c(0)
head(new.daily)

# make data frame to save predictions from each model for each shop
shop_preds <- data.frame(shop=c(0:59),
                         pred=c(0))

# for loop (explained in further detial in the paper)
for (i in 0:59) {
  
  shop_df <- sales1[which(sales1$shop_id==i),]
  merged <- merge(new.daily, shop_df,
                  by=c("year", "month", "day"),
                  all.x = TRUE)
  merged[is.na(merged)] = 0
  merged1 <- merged[,c(1:3, 6)]
  
  merged.ts <- ts(merged1[,4],
                   frequency=365,
                   start(2013, 1))
  
  knn.merged <- knn_forecasting(merged.ts, h=30, lags=1:365,
                                 k=35, msas="recursive", cf="median")
  
  pred.nov <- sum(knn.merged$prediction)
  shop_preds$pred[i+1] <- pred.nov
  
}

# show predictions by shop
shop_preds
# save sum of predictions (this is total for month of Nov)
sum_shops <- sum(shop_preds$pred)


# make a model with the optimum parameters to forecast november (not by shop)
all_daily.ts <- ts(daily.sales[,4],
               frequency=365,
               start(2013, 1))
knn.d.tuned2 <- knn_forecasting(all_daily.ts, h=30, lags=1:365, 
                                k=35, msas="recursive", cf="median")
# save preds
pred.da.nov <- sum(knn.d.tuned2$prediction)

# compare prediction from different models (by shop vs in total)
pred.da.nov
sum_shops


# use rolling origin to understand accuracy of model
rolling <- rolling_origin(knn.merged, h=30)
# look at test sets
rolling$test_sets
# loook at prediction by training set to compare to test set
rolling$predictions
# look at error (test set vs preds)
rolling$errors
# global accuracy
rolling$global_accu
# accuracy by forecasting horizon
rolling$h_accu
# plot the rolling origin
plot(rolling, h=30)

# compare to global accuracy from first (not tuned) model
knn.merged.2 <- knn_forecasting(merged.ts, h=30, lags=1:365,
                              k=25, msas="recursive", cf="mean")
rolling2 <- rolling_origin(knn.merged.2)
rolling2$global_accu
