Predicting the total sales for the month of November using the dataset obtained from Kaggle.

The problem that we decided to approach was to predict the future product sales 
of the 1C Company for the month of November. With this predict or forecast problem, 
there are many methods that can be utilized. In our case, we utilized the K-Nearest Neighbor 
and Random forest. First, we use k=35 and k=25 to establish two K-Nearest Neighbor models. 
When k=35, the accuracy measures show a MAPE of 46.38%. Similarly, when k=25, 
the accuracy measures show a MAPE of 63.63%. 

After calculating the accuracy and mape for the shops, we noticed that there are some shops 
that have an irregular value. This would mean that those specific shops did not have enough 
data to fit our lag parameters. Otherwise, the accuracies for shops that do have decent percentages, 
which means that our models did work under certain conditions. For example, shop 2 had a prediction of 
860.48, while shop 4 had 848.85. The mape for shop 2 and 4 was 16.91% and 13.18%, respectively. 
For accuracy, shop 2 had 83.09%, while shop 4 had 86.82%. The two accuracies were good and it means 
that our model was able to predict for these two shops. 

Therefore, in general, the K-Nearest Neighbor method is the most suitable for modeling
since our dataset is a time series and the approach is really known to be used mostly for 
classification and regression problems. K-Nearest Neighbor mainly relies on the surrounding 
limited nearby samples, rather than the method of discriminating the class domain bai to determine
the category. Therefore, for the sample set du that has more cross or overlap of the class domain, 
the K-Nearest Neighbor method is more suitable than other methods.

In the K-Nearest Neighbor model, if we use the bottom-up approach,
we will get the total sales forecast for November to be 66599.
If we change to a top-down method and use overall daily sales data to predict sales in November,
the total sales value in November is 77,116, which is an increase of 10,517 over our bottom-up method. 
