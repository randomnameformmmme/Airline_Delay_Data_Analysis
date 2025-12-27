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

### Running the Analysis
```python
# Load the dataset
import pandas as pd
df = pd.read_csv('Airlines.csv')

# Run preprocessing
# (Include your preprocessing steps)

# Execute analysis
# (Include your analysis steps)
```

## Key Insights

1. **Airline Impact**: The choice of airline is the strongest predictor of flight delays
2. **Temporal Patterns**: Evening and afternoon flights are significantly more prone to delays
3. **Weekly Trends**: Mid-week flights (especially Wednesday) show higher delay rates
4. **Flight Duration**: Operational complexity in longer flights contributes to higher delays

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
