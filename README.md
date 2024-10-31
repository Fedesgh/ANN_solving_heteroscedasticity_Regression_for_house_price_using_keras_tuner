Using a dataset of house prices from (https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
We aim to build an artificial neural network (ANN) optimized with Keras-Tuner.

♦About the data set: 

Number of rows: 21613

Number of columns: 21

We use 18 columns as features (inputs), 1 as target (continuous variable "price"), and drop 2

## Model results

♦Structure:


![model.ss.jpg](model.ss.jpg)


♦Hyperparameters:


num_layers: 6

units_0: 63

activation: leaky_relu

dropout: False

lr: 0.00110654178783952





♦Metrics: 


MAE:  71602.69808292852


MAPE:  0.1280146931732677


![scatterplot.jpg](scatterplot.jpg)


♦ The results show signs of **heteroscadesticity** so we perform a **Breusch-Pagan Test**


![pvalue.jpg](pvalue.jpg)


Wich confirms it.

## Trying to solve **heteroscadesticity**y. 

There are several ways to overcome **heteroscadesticity**, one of them is to try to transform the values ​​of the features that did not work in this case. 


Another solution may be to use a weighted least squares (WLS) model that we developed (WLS.ipybn) using the **statsmodels** library.


First we need to get **MSE** from the trained **OLS** model so that we can use it to assign **weights** as parameter in **WLS**.


♦Results metrics: 


![images/metrics.jpg](images/metrics.jpg)


Although **WLS** performs better than **OLS** as it is supposed to do in this case, both models perform poorly compared to our **ANN** with MAE: 71602.69808292852.2.



![images/OLSv.jpg](images/OLSv.png)


## Attempt at segmentation through Kmeans 



The idea is to build an unsupervised model to segment the data before applying a regression model to predict the price. Hopefully, we will be able to differentiate complicated data sets from simple ones. We train an optimal Kmeans with K=5 which shows us that there are 2 groups of 5, with a wider price range.


![images/Segmentation/Segments.jpg](images/Segmentation/Segments.jpg)


![images/Segmentation/Seglabels.jpg](images/Segmentation/Seglabels.jpg)


Then we separate the data in labels **[2,4]** and **[0,1,3]** and build an **ANN** for each group 


### ANN for labels [0,1,3]


♦Structure:


![images/Segmentation/estructure0,1,3.jpg](images/Segmentation/estructure0,1,3.jpg)


♦Metrics:


![images/Segmentation/metric0,1,3.png](images/Segmentation/metric0,1,3.png)


♦ MAE:  50053.131876179614


♦ MAPE:  0.12146668562655259


### ANN for labels [2,4]


♦Structure:


![images/Segmentation/structure2,4.jpg](images/Segmentation/structure2,4.jpg)


♦Metrics:


![images/Segmentation/metric2,4.png](images/Segmentation/metric2,4.png)


♦MAE:  122800.32979130244


♦MAPE:  0.14176357064716935

## Quantile attempt 

We treat **heteroscedasticity** as **outliers**


First sort the data and graph .95 quantile:


![images/quantile/Quantilegra.png](images/quantile/Quantilegra.png)

### Build an ANN classifier to detect whether new data belongs outside (1) or inside (0) the 0.95 quantilee


♦Structure:


![images/quantile/StructureClassQuantile.jpg](images/quantile/StructureClassQuantile.jpg)



♦Metrics


![images/quantile/MetricsClassQuantile.png](images/quantile/MetricsClassQuantile.png)


♦MAE:  66826.45548567994


♦MAPE:  0.14593678552121173


### Training ANN regression for .95 quantile

♦Structure:


![images/quantile/Structure95reg.jpg](images/quantile/Structure95reg.jpg)


♦Metrics:


![images/quantile/Metrics95.png](images/quantile/Metrics95.png)


♦MAE:  66826.45548567994


♦MAPE:  0.14593678552121173


### Training an ANN regression for .05 


♦Structure:


![images/quantile/structure05.jpg](images/quantile/structure05.jpg)


♦Metrics:


![images/quantile/metrics05.png](images/quantile/metrics05.png)


♦MAE:  287107.61664746545


♦MAPE:  0.16901369037863703


# Final and most important step:

We classify all the data through our **ANN classifier** to predict whether they belong to the .95 quantile or not. 

After classification we apply the correct **ANN regression** for their classes

♦Results: 


**For data classified as 0.95 quantile**: 


MAE:  42557.6068662773


MAPE:  0.09224515971000753

**For data classified as 0.05 **:


MAE:  171365.52690029616


MAPE:  0.09833767941673963


## Foot note
I noticed that there is an issue with the WLS formula in the statsmodels documentation.(https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.WLS.html)


It is not clear whether we should apply **w^2** or **w**, the first option works better in our **WLS**


There is an open forum on the case:

(https://stackoverflow.com/questions/46000839/is-the-example-of-wls-in-statsmodels-wrong)










