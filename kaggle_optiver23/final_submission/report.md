## REPORT ON KAGGLE "TRADING AT THE CLOSE"

### Label and Evaluation

In this competition, the challenge is to develop a model capable of **utilizing order book and closing auction data to predict the closing price trends of hundreds of NASDAQ-listed stocks**. Information from the auctions can be used to adjust prices, assess supply and demand dynamics, and identify trading opportunities. The model can help integrate signals from auctions and order books, thereby enhancing market efficiency and accessibility, particularly in the tense final ten minutes of trading.

The `target` is defined as **the difference in the 60-second future change in the weighted average price (WAP) of the stock and the 60-second future change in the WAP of a synthetic index**. The unit of the target is basis points, a common unit of measurement in financial markets where a 1 basis point price move corresponds to a 0.01% price move.

$$\text{Target} = \left( \text{StockWAP}_{t+60} / \text{StockWAP}_t - \text{IndexWAP}_{t+60} / \text{IndexWAP}_t \right) \times 10000$$

Submissions are evaluated on the **MAE** between the predicted return and the observed target. 

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - x_i|$$

### Background Information

![Screenshot 2024-04-11 at 12.40.29 AM](/Users/kevinshuey/Desktop/Screenshot 2024-04-11 at 12.40.29 AM.png)

### Dataset Description

This dataset provides detailed historical data for the daily ten-minute closing auctions on the NASDAQ stock exchange, aiming to challenge participants to predict the future price movements of stocks relative to a synthetic index composed of NASDAQ-listed stocks. The data includes various **features for both individual stocks and the synthetic index, such as prices, trading volume, imbalances**, etc. Additionally, labels are provided in the training set representing the price movements of stocks relative to the synthetic index. The goal of the challenge is to build models that accurately predict the future movements of stock prices relative to the synthetic index, thereby aiding investors in making more informed trading decisions.

### General pipeline

**Feature Engineering:**

Statistical features: Calculating **group statistics** like mean, standard deviation, median, minimum, and maximum.

Count features: Determining the **count** and **the number of unique values within groups**.

**Model Training:**

Algorithms used include LightGBM, XGBoost, CatBoost, along with deep learning architectures like DNN, Transformer, and GRU.

The technique of **stacking** is also employed, which is a method of combining multiple models.

**Validation Strategy:**

Group **K-fold cross-validation**.

**Time-based splitting** for training and validation sets

**Dataset Information:**

The training dataset consists of IDs ranging from 0-480, and there's an additional test dataset with IDs from 0-199, including 200 additional features.

### Possible features engineering selection:





