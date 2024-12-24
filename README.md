# Real Estate Price Discovery and Regulation Model

## Project Overview
This project develops a machine learning model for accurate real estate price prediction in Pune, Maharashtra. It aims to build trust through transparent and accurate property pricing using advanced data analytics and machine learning techniques.

## Business Problem
- Company X operates a nationwide real estate aggregator platform
- Users report significant price variations for similar properties
- Need for accurate and transparent pricing to maintain trust and market leadership

## Features
- Property price prediction based on multiple features
- Outlier detection and handling
- Model comparison and evaluation
- Confidence interval calculations for price ranges
- Interactive visualizations

## Tech Stack
- Python 3.x
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - lightgbm
  - catboost
  - matplotlib
  - seaborn

## Project Structure
```
pune_real_estate/
├── data/
│   ├── raw/             # Original data files
│   └── processed/       # Cleaned and processed data
├── notebooks/          # Jupyter notebooks for exploration
├── src/               # Source code
│   ├── data/          # Data processing scripts
│   ├── features/      # Feature engineering
│   ├── models/        # Model training and evaluation
│   └── visualization/ # Data visualization
├── reports/          # Generated analysis reports
│   └── figures/      # Generated graphics
└── requirements.txt  # Project dependencies
```

## Models Tested
1. Gradient Boosting Models:
   - XGBoost
   - LightGBM
   - CatBoost
2. Linear Models:
   - Lasso Regression
   - Ridge Regression
3. Neural Networks:
   - MLP (Multi-Layer Perceptron)

## Model Performance
Best Model: LightGBM
- Training RMSE: 10.04 lakhs
- Test RMSE: 10.84 lakhs
- R² Score: 0.9458

## Installation & Usage

1. Clone the repository:
```bash
git clone https://github.com/YourUsername/pune-real-estate-prediction.git
cd pune-real-estate-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the main script:
```bash
python src/main.py
```

## Data Requirements
The model expects an Excel file with the following features:
- Property Area in Sq. Ft.
- Property Type (BHK)
- Location details
- Amenities information
- Price in lakhs (target variable)

## Deployment Strategy
- Three-stage deployment pipeline (Development, Staging, Production)
- Comprehensive monitoring system
- Automated CI/CD implementation using GitHub Actions
- Integration with MLflow for experiment tracking

## Results and Visualizations
The project generates:
- Model comparison reports
- Price prediction intervals
- Feature importance analysis
- Distribution plots
- Performance metrics visualization

## Recommendations
1. Create interactive dashboards for real-time insights
2. Schedule bi-weekly data-driven presentations
3. Implement user feedback system
4. Collect more data before high-stakes deployment
5. Scale dataset with external sources

## Contributor
- Leonidas Mugema
