# Airline Delay Prediction Analysis

A comprehensive data analysis and machine learning project focused on predicting airline delays using classification models. This project analyzes over 539,000 flight records to identify key factors contributing to flight delays.

## Project Overview

This project performs exploratory data analysis and develops machine learning models to predict flight delays. Through statistical hypothesis testing and classification algorithms, we identify operational factors that significantly impact flight punctuality.

### Key Findings

- **Airlines & Airports**: Major airlines and airports show varying delay patterns
- **Time Patterns**: Evening flights have the highest delay rate (51.38%), closely followed by afternoon flights (50.19%)
- **Day of Week**: Wednesday shows the highest delay rate (47.08%), contrary to initial expectations
- **Flight Length**: Longer flights demonstrate higher delay rates, with "Long" flights showing the highest delays (44.87%)

## Research Hypotheses

**H1**: Major airlines and airports would have more delayed flights than others
- **Result**: Validated through statistical analysis

**H2**: Sunday evening flights would have the highest delay rate
- **Result**: Rejected - Wednesday showed the highest delay rate, though evening time period was confirmed

**H3**: Longer flights have higher delay rates due to operational complexity
- **Result**: Confirmed - Long flights showed highest delays, though "Very Long" flights showed slightly lower rates

## Dataset

- **Source**: Kaggle Airlines Dataset
- **Records**: 539,383 flights
- **Features**: 9 initial columns (expanded through feature engineering)
- **Target Variable**: Delay status (On-time vs. Delayed)

### Original Features
- Airline
- Flight number
- Airport From/To
- Day of Week
- Departure Time
- Flight Duration
- Delay Status

### Engineered Features
- `DepartureFormatted`: Time in HH:MM format (00:00-23:59)
- `DurationFormatted`: Duration in Xh XXm format
- `departure_tp`: Time period categories (Early Morning, Morning, Afternoon, Evening)
- `flight_length`: Flight categories (Short, Medium, Long, Very Long)
- `delay_status`: Categorical delay status (On-time/Delayed)

## Project Structure

```
├── Airline_Delay_Data_Analysis__3_.ipynb    # Main analysis notebook
├── decisiontreeds.py                         # Decision Tree classifier
├── kNN_Model_Airline.py                      # k-Nearest Neighbors classifier
├── Random_Forest.py                          # Random Forest classifier
├── Airlines.csv                              # Original dataset
└── Modified_Airlines.csv                     # Processed dataset with engineered features
```

### File Descriptions

**Airline_Delay_Data_Analysis__3_.ipynb**
- Complete exploratory data analysis workflow
- Data cleaning and preprocessing
- Feature engineering implementation
- Statistical hypothesis testing
- Chi-square analysis for categorical variables
- Data visualization and insights generation

**decisiontreeds.py**
- Decision Tree classifier implementation
- Uses entropy criterion for splitting
- Features: DayOfWeek, departure_minutes, duration_minutes, flight_length
- Model parameters: max_depth=3, min_samples_leaf=5
- Outputs: confusion matrix, accuracy score, classification report

**kNN_Model_Airline.py**
- Custom k-Nearest Neighbors implementation from scratch
- k=3 neighbors configuration
- Includes data standardization using StandardScaler
- Visualizations: confusion matrix heatmap, performance metrics bar chart
- Metrics: accuracy, precision, recall, F1-score
- Note: Uses subset of data (10,000 rows) for computational efficiency

**Random_Forest.py**
- Random Forest ensemble classifier
- Comparison with Decision Tree performance
- Features: DayOfWeek, departure_minutes, duration_minutes, flight_length, departure_tp
- Model parameters: n_estimators=100, max_depth=None
- Custom threshold adjustment (0.45) for class prediction
- Feature importance analysis included

## Methodology

### Data Preprocessing
1. **Data Cleaning**
   - Removed unnecessary columns (ID)
   - Verified no missing or duplicate values
   - Validated data integrity

2. **Feature Engineering**
   - Reformatted time columns for better interpretability
   - Created categorical bins for departure times and flight lengths
   - Generated human-readable status labels

3. **Exploratory Data Analysis**
   - Statistical summaries using `describe()`
   - Distribution analysis across key variables
   - Correlation analysis between features

### Statistical Analysis
- **Chi-Square Tests**: Evaluated relationships between categorical variables and delay status
- **Significance Level**: α = 0.05
- **Factors Analyzed**:
  - Airline (strongest association)
  - Time period of departure
  - Flight length
  - Day of the week

All factors showed statistically significant relationships with delay status (p < 0.05).

### Machine Learning Models
- **k-Nearest Neighbors (k-NN)**
- **Decision Tree Classifier**
- **Random Forest Classifier**

## Results Summary

| Factor | Chi-Square Statistic | P-Value | Significance |
|--------|---------------------|---------|--------------|
| Airline | Highest | < 0.05 | ✓ Strong |
| Departure Time Period | High | < 0.05 | ✓ Strong |
| Flight Length | Moderate | < 0.05 | ✓ Moderate |
| Day of Week | Lower | < 0.05 | ✓ Significant |

### Delay Rates by Time Period
- Evening: 51.38%
- Afternoon: 50.19%
- Morning: Lower
- Early Morning: Lowest

### Delay Rates by Flight Length
- Long: 44.87%
- Very Long: 44.87%
- Medium: Lower
- Short: Lowest

## Technologies Used

- **Python 3.x**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning models
- **Matplotlib/Seaborn**: Data visualization
- **Google Colab**: Development environment

## Installation & Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Dataset Setup
1. Download the Airlines dataset from [Kaggle](https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay)
2. Place `Airlines.csv` in your project directory

### Running the Analysis

#### 1. Data Preprocessing and EDA
Run the Jupyter notebook for complete exploratory analysis:
```bash
jupyter notebook Airline_Delay_Data_Analysis__3_.ipynb
```

This notebook will:
- Load and inspect the raw dataset
- Perform data cleaning and validation
- Engineer new features
- Conduct statistical hypothesis testing
- Generate the `Modified_Airlines.csv` file for model training

#### 2. Decision Tree Model
```bash
python decisiontreeds.py
```
Update the file path in the script:
```python
df = pd.read_csv('/path/to/your/Airlines.csv')
```

#### 3. k-Nearest Neighbors Model
```bash
python kNN_Model_Airline.py
```
Update the file path in the script:
```python
csv_file = "Modified_Airlines.csv"
```
Note: Default uses 10,000 rows for faster execution. Remove `df.head(10000)` to use full dataset.

#### 4. Random Forest Model
```bash
python Random_Forest.py
```
Update the file path in the script:
```python
df = pd.read_csv('/path/to/your/Modified_Airlines.csv')
```

### Model Outputs

All models provide:
- Confusion matrix
- Accuracy score (%)
- Classification report (precision, recall, F1-score)

Additional outputs:
- **kNN**: Visualization plots for confusion matrix and performance metrics
- **Random Forest**: Feature importance rankings

## Model Performance

### Classification Models Comparison

**Decision Tree Classifier**
- Criterion: Entropy
- Max Depth: 3
- Min Samples per Leaf: 5
- Features Used: 4 (DayOfWeek, departure_minutes, duration_minutes, flight_length)

**k-Nearest Neighbors (k-NN)**
- k Value: 3
- Distance Metric: Euclidean
- Preprocessing: StandardScaler normalization
- Features Used: All available features (one-hot encoded categorical variables)
- Computational Note: Demonstrated on 10,000 sample subset

**Random Forest Classifier**
- Number of Estimators: 100
- Max Depth: None (unlimited)
- Min Samples per Leaf: 1
- Features Used: 5 (DayOfWeek, departure_minutes, duration_minutes, flight_length, departure_tp)
- Custom Prediction Threshold: 0.45
- Class Balancing: Applied to handle imbalanced dataset

### Feature Importance (Random Forest)
The Random Forest model provides feature importance rankings, identifying which operational factors contribute most to delay predictions. Key features are ranked by their impact on model decisions.

## Key Insights

1. **Airline Impact**: The choice of airline is the strongest predictor of flight delays
2. **Temporal Patterns**: Evening and afternoon flights are significantly more prone to delays
3. **Weekly Trends**: Mid-week flights (especially Wednesday) show higher delay rates
4. **Flight Duration**: Operational complexity in longer flights contributes to higher delays

## Technical Implementation Notes

### Data Processing Pipeline
1. **Initial Data Loading**: Raw CSV import with 539,383 records
2. **Feature Engineering**: Creation of 5 new features for improved interpretability
3. **Categorical Encoding**: Conversion of categorical variables to numerical codes for model compatibility
4. **Train-Test Split**: 70-30 split with random_state=100 for reproducibility

### Code Organization
- **Preprocessing Code**: Centralized in Jupyter notebook, reused across all model files
- **Modular Design**: Each model in separate file for independent execution
- **Consistent Evaluation**: All models use identical metrics for fair comparison

### Performance Considerations
- **kNN Memory Optimization**: Subset sampling (10,000 rows) to manage computational resources
- **Random Forest Threshold Tuning**: Custom 0.45 threshold to optimize precision-recall tradeoff
- **Class Imbalancing Handling**: Balanced class weights in Decision Tree and Random Forest

### Reproducibility
All models use `random_state=100` for consistent results across runs. The same train-test split ensures fair model comparison.

## Future Work

- Incorporate weather data for enhanced prediction accuracy
- Analyze seasonal patterns in flight delays
- Develop deep learning models for improved performance
- Create interactive dashboards for real-time delay prediction
- Expand analysis to include aircraft manufacturer data

## Project Team

**CISC 5380-02: Programming with Python**  
Fall 2025, Fordham University

## License

This project is part of academic coursework at Fordham University.

## Acknowledgments

- Dataset source: Kaggle Airlines Dataset
- Course Instructor: Dylan Smith
- Fordham University Data Science Program

---

**Note**: This analysis was conducted as part of a graduate-level course in data analytics. The dataset was pre-cleaned for machine learning purposes, ensuring data quality and reliability.
