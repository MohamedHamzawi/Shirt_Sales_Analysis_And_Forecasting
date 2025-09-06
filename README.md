# Shirt Orders Sales Analysis And Forecasting

This project covers the analysis and forecasting of shirt order sales. The analyis of the data aims to derive insights on revenue trends, customer preferences, and order distributions. The analysis spans data loading, cleaning, merging, and exploratory data analysis.
The forecasting aims to predict the number of shirt sales for the next specific number of days. The forecasting encorprates a time-series machine-learning forecasting model called ARIMA. The model is then evaluated using a couple of metrics and monitored using mlflow. Then the model was deployed using streamlit

 # Data Processing Steps
### Loading Data: Data is loaded from three sources: customer, product, and orders databases.
### Merging Data: The datasets are merged to form a single comprehensive dataset.
### Cleaning Data: The dataset is cleaned by removing unnecessary fields and calculating relevant metrics like revenue considering discounts.
### Feature Engineering: New columns such as year, month, and day are extracted from OrderDate for more detailed analysis.
# Exploratory Data Analysis
### Annual Trends: Visualization of yearly revenue and orders to identify growth trends.
### Monthly and Daily Revenue: Detailed analysis of revenue trends by month and day to pinpoint peak performance periods.
### Geographical Distribution: Analysis of revenue by state to understand regional market penetration.
### Product and Customer Insights: Identification of top-selling products and highest revenue-generating customers.
# Visualizations
## The Power BI dashboard presents a visual summary of key metrics and trends:

### Revenue and Orders by Time: Insights into how sales performance varies by month, day, and year.
### Customer Demographics: Breakdown of orders by gender.
### Geographical Insights: State-wise revenue distribution.
### Key Performance Indicators: Snapshots of total revenue, total orders, average discounts, and total units ordered.
# Conclusion
### This analysis provides a comprehensive view of sales performance, helping stakeholders make informed decisions about marketing strategies, product placement, and customer engagement.

# Data and Resources
### Data sources include internal company records of orders, products, and customer demographics. The analysis is performed using Python for data processing and Power BI for visualization

# Time series forecasting
### Decided to use the ARIMA (AutoRegressive Integrated Moving Average) model to forecast sales for the next two months based on the past data. The ARIMA model was chosen for its ability to handle non-stationary time series and account for sales trends and seasonality. The model was trained on historical shirt sales data, and the results showed promising accuracy in predicting short-term future sales. Fine-tuned the ARIMA model by testing different parameter configurations, optimizing for better predictive performance.

# Model tracking and deployment
### Logged the model's R-squared and MSE metrics to track how well the model fits the data. Then integrated MLflow to track and manage multiple versions of the forecasting model. Logged model parameters, metrics, and artifacts for version control and easy access for future comparisons. Utilized MLflow’s tracking server to monitor experiment results over time and ensure reproducibility. Deployed the ARIMA forecasting model using Streamlit, a lightweight web application framework, allowing users to interact with the model.
