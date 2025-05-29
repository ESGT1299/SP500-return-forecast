# S&P 500 Next-Day Return Forecast Using Machine Learning and Deep Learning

This portfolio project (by Erick Guagua, Physics B.Sc.) aims to forecast the next-day returns of the S&P 500 index using advanced machine learning (ML) and deep learning (DL) methods. The goals are to engineer informative features from historical price data, evaluate a range of models (e.g. random forests, XGBoost, neural networks), and deploy an interactive Streamlit dashboard for visualization and real-time forecasting. This approach could help financial analysts and quants anticipate market movements and inform trading or hedging strategies, while acknowledging the inherent unpredictability of markets

## Project Structure

sp500-return-forecast/
├── data/
│ ├── raw/
│ └── processed/ 
├── notebooks/
│ ├── 01_download.ipynb
│ ├── 02_preprocessing.ipynb
│ ├── 03_baselines.ipynb
│ ├── 04_deep_learning.ipynb
│ ├── 05_evaluation.ipynb
│ ├── 06_improved_models.ipynb
│ ├── 07_deep_learning_improved.ipynb
│ ├──08_ensemble_validation.ipynb
│ └──09_deploy_streamlit.py 
├── models/
├── requirements.txt
├── .gitignore
└── README.md

## Methodology

1. **Data & Features:** We use historical S&P 500 price data (daily OHLCV) as input. Feature engineering includes classic technical indicators (e.g. moving averages, momentum oscillators like RSI, MACD) and lagged returns to capture momentum and mean-reversion effects. These features supplement raw price inputs to help models detect patterns. (As one study notes, indicators such as SMA, MACD, and RSI can power high-accuracy LSTM forecasting models). We also consider market sentiment or macro variables if available, though the core features come from price history and volume.

2. **Model Selection:** A diverse set of models is evaluated. Classical ML models (scikit-learn implementations of linear regression, support vector machines, random forests) provide baselines. We include XGBoost, a gradient-boosted tree ensemble known for state-of-the-art performance in many challenges, and neural network architectures (e.g. LSTM recurrent nets or feedforward nets implemented in TensorFlow/Keras). TensorFlow (with its high-level Keras API) is used for DL models, as it “makes it easy for beginners and experts to create machine learning models” on various platforms. We train all models on historical data using cross-validation and tune hyperparameters via grid or random search.

3. **Ensemble Methods:** To improve robustness, we implement ensemble techniques. By aggregating multiple predictive models, we can reduce variance and improve accuracy. For example, we blend or stack the outputs of XGBoost, random forest, and neural net regressors, weighting them based on validation performance. Such ensembles have been shown to outperform individual models in stock return prediction tasks. Feature importance from tree-based models (e.g. XGBoost) also guides feature selection.

4. **Evaluation:** Models are evaluated on historical hold-out data using error metrics like Mean Absolute Error (MAE) and directional accuracy (up/down movement). We compare predictive accuracy against naive benchmarks (e.g. yesterday’s close as forecast). As reported in the literature, advanced models (especially XGBoost) often yield low percentage errors (~3–5%) and significant predictive power in similar tasks. However, we remain cautious: stock returns are noisy and any forecast has uncertainty (see Caveats below).

5. **Streamlit Deployment:** To make the models accessible, we build a Streamlit dashboard. Streamlit is an open-source Python framework that “enables [data scientists and AI/ML engineers] to deliver dynamic data apps with only a few lines of code”. In practice, the Streamlit app allows users to select a date or input features and instantly see the predicted S&P 500 return for the next day, along with interactive charts of historical data. No web development skills are required (Streamlit apps are “shareable web apps in minutes” using pure Python).


## Results & Outcomes

## Results & Outcomes

| Model                     | Configuration                     | Hold-out MAE | Hold-out R² |
|:--------------------------|:----------------------------------|-------------:|------------:|
| ARIMA                     | default parameters                | 0.01103      | 0.001       |
| Random Forest             | default scikit-learn              | 0.00785      | 0.128       |
| MLP                       | basic (no tuning)                 | 0.11887      | −234.074    |
| MLP                       | tuned (Keras-Tuner best)          | 0.00707      | 0.384       |
| LSTM                      | basic architecture                | 0.01439      | −4.157      |
| LSTM                      | improved/stacked (two layers)     | 0.01217      | −1.319      |
| **Ensemble**              | stack of RF + XGB + MLP + LSTM    | **0.00751**  | **0.095**   |
| **Ensemble (walk-forward)** | 5-fold rolling window average   | 0.00758 ± 0.00226 | 0.099 ± 0.045 |


> **MAE**: Mean Absolute Error of log-returns (lower is better).  
> **R²**: Coefficient of determination (closer to 1 indicates stronger explanatory power).

## Real-World Use Cases

1. **Quantitative Trading:** Hedge funds and quantitative traders can incorporate these forecasts as one input in their algorithmic strategies. For example, a short-term trading desk might increase exposure when the model predicts a positive return, or hedge risk when a decline is forecast. Predictive analytics is now essential in finance: it “enables [funds] to decipher complex market dynamics and anticipate future trends more precisely, identifying potential investment opportunities and risks”. Machine learning models can spot subtle patterns in price and volume data that traditional analysis might miss. 

2. **Portfolio Management and Risk:** Portfolio managers can use next-day return forecasts to adjust allocations or hedge positions. For instance, if the model signals a likely downturn, a manager might reduce equity holdings or buy protective options. The literature notes that such ML-based forecasts “help traders, investors, as well as portfolio managers better predict stock market trends and, in turn, the returns”. In practice, these predictions are one of many signals (alongside fundamentals and macro data) that inform strategy.

3. **Retail/Institutional Platforms:** Financial platforms or brokerage dashboards could integrate the Streamlit app to provide traders with AI-driven insights. By visualizing predictions in real-time, both retail and institutional users can benefit. (Note that actual trading decisions should consider transaction costs and confirm signals with additional analysis.)

## Installation and Usage

Follow these steps to run the project locally and launch the Streamlit dashboard:

1. **Clone the repository:**
```bash
git clone https://github.com/ESGT1299/sp500-return-forecast
cd sp500-forecast-ml
```
2. **Install dependencies:**
Ensure Python 3.8+ is installed. Then create a virtual environment (optional) and install required packages:
```bash
pip install -r requirements.txt
```
The requirements.txt includes key libraries such as scikit-learn (ML algorithms and utilities), TensorFlow (for deep learning), XGBoost, pandas, numpy, and Streamlit.

3. **Data setup:**

By default, the project uses historical S&P 500 data from Yahoo Finance (via yfinance) or an equivalent source. If an API key is required, obtain one (e.g. from AlphaVantage) and set it in the configuration or environment variables. Ensure the data file (e.g. data/sp500.csv) is present

4. **Training models:**

Run the training pipeline to preprocess features, train models, and save the best models. This script will output performance metrics and store trained model files in the models/ directory. (You can adjust hyperparameters or use cross-validation by editing the script.)

5. **Launching the Streamlit app:**
Start the interactive dashboard with:
```bash
streamlit run app.py
```

By default, Streamlit runs at http://localhost:8501. In the web interface, you can select input parameters or dates and view the predicted next-day return, along with plots of historical prices and feature importance. The dashboard code automatically loads the trained models.

6. **Usage Notes:**

Modify requirements.txt or the Streamlit config if you need to change ports or enable remote access. To update data, rerun the data download script and retrain the models.

## Key Tools and Libraries

**Scikit-learn:** A versatile Python ML library offering many classifiers and regressors (e.g. RandomForestRegressor, SVR).

We use scikit-learn for preprocessing (scaling, train-test splits, etc.) and some baseline models.

**XGBoost:** A scalable gradient boosting framework that achieves state-of-the-art results in many domains.

We leverage XGBoostRegressor for its efficiency and accuracy in regression tasks.

**TensorFlow / Keras:** We use TensorFlow (with Keras API) for building neural networks (e.g. LSTM or dense layers). TensorFlow’s high-level Keras interface is “designed for human beings” and simplifies model development. Its ecosystem supports efficient training on GPUs if available.

**Streamlit:** An open-source Python framework to quickly build and share data apps. With Streamlit we turn the forecasting code into an interactive web app (graphs, sliders, and tables) without having to write HTML/JS. 

## Caveats and Future Work

**Forecast Uncertainty:** Financial markets are notoriously hard to predict. According to the Efficient Market Hypothesis (EMH), stock prices reflect all available information, making consistent alpha generation extremely challenging. In practice, our models may occasionally capture short-term patterns, but forecasts should not be treated as guarantees. Model performance can degrade in regime shifts (e.g. sudden crashes or rallies). Users must exercise caution and not rely solely on these predictions for trading.

**Data and Model Limitations:** The project currently uses only historical price/volume data and technical indicators. Future improvements could incorporate alternative data (news sentiment, macroeconomic variables, social media) to enrich features. Additionally, more sophisticated architectures (like transformer-based time series models) or ensembling methods (stacking meta-learners) could be explored. We also assume that daily returns are a suitable target – one could extend to directional classification (up/down) or even high-frequency intraday predictions.

**Overfitting and Robustness:** Given limited data relative to model complexity, overfitting is a risk. Cross-validation and regularization help mitigate this, but out-of-sample testing in live conditions is essential. Future work should include walk-forward testing and stress-testing models under different market conditions.

**Maintenance:** The Streamlit app currently uses static models. For production use, one would automate data updates and model retraining on new data. Real-time integration (e.g. using APIs or Streamlit Cloud) could make the forecasts continuously current.

## References

1. Achelis, S. B. (2001). *Technical Analysis from A to Z*. McGraw-Hill.

2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM. Available at https://arxiv.org/abs/1603.02754

3. Chollet, F. (2018). *Deep Learning with Python*. Manning Publications.

4. Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *The Journal of Finance, 25*(2), 383–417.

5. Mohapatra, S., Mukherjee, R., Roy, A., Sengupta, A., & Puniyani, A. (2022). Can ensemble machine learning methods predict stock returns for Indian banks using technical indicators? *Journal of Risk and Financial Management, 15*(8), 350. https://doi.org/10.3390/jrfm15080350

6. Phuoc, T., Thi Kim Anh, P., Phan Huy Tam, P., & Nguyen, C. V. (2024). Applying machine learning algorithms to predict the stock price trend in the stock market – the case of Vietnam. *Humanities and Social Sciences Communications, 11*, 393. https://doi.org/10.1057/s41599-024-01393-6

7. Polikar, R. (2006). Ensemble based systems in decision making. *IEEE Circuits and Systems Magazine, 6*(3), 21–45. https://doi.org/10.1109/MCAS.2006.1688199

8. Streamlit Inc. (2022). *Streamlit Documentation*. Retrieved from https://docs.streamlit.io

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for bug fixes, enhancements, or new features.

## License
This project is released under the MIT License.
