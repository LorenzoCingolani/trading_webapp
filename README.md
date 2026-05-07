# Trading Web Application

A comprehensive Streamlit-based trading analysis platform designed to perform quantitative market analysis using multiple trading strategies and generate actionable trading signals. The application processes market data through various strategies (EWMA, Carry, Breakout, Stochastic) and produces outputs for portfolio management and order execution.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [How to Run](#how-to-run)
- [Application Workflow](#application-workflow)
- [Trading Strategies](#trading-strategies)
- [Data Flow](#data-flow)
- [Future Enhancements](#future-enhancements)

## Features

1. **Multi-Strategy Analysis**: Implements EWMA, Carry, Breakout, and Stochastic trading strategies
2. **Interactive Web Interface**: Streamlit-based UI with multiple analysis pages
3. **Automated Validation**: Validates strategy outputs and correlations
4. **Portfolio Diversification Multiplier (PDM)**: Calculates optimal position sizing based on portfolio diversification
5. **Combined Forecasting**: Aggregates signals from multiple strategies
6. **Sharpe Ratio Analysis**: Evaluates risk-adjusted returns for each strategy
7. **Order Generation**: Creates formatted order files for execution

## Project Structure

```
trading_webapp/
├── app.py                      # Main Streamlit application entry point
├── DATA/
│   ├── input_instruments/      # Input CSV files for each trading instrument
│   ├── output_instruments/     # Strategy outputs (EWMA, Carry, etc.)
│   ├── combinedForecast/       # Combined strategy forecasts
│   ├── order_folder/           # Generated order files
│   └── output_plots/           # Visualization outputs
├── p_pages/                    # Streamlit page modules
│   ├── main_analysis_page.py   # Strategy execution page
│   ├── validation_page.py      # Output validation and correlation analysis
│   ├── pdm_page.py            # Portfolio Diversification Multiplier
│   ├── forecast_page.py        # Combined forecast generation
│   └── sharpe_ratio_page.py    # Risk-adjusted performance metrics
├── steps/                      # Core analysis functions
│   ├── p1_analysis.py         # Main strategy execution logic
│   ├── p2_validation.py       # Validation logic
│   ├── p3_pdm.py              # PDM calculations
│   └── p6_sharpe_ratio.py     # Sharpe ratio calculations
├── strategies/                 # Strategy implementations
│   ├── ewma.py                # Exponentially Weighted Moving Average
│   ├── carry.py               # Carry strategy (commodity/FX)
│   ├── break_model.py         # Breakout strategy
│   └── stochastic_breakout.py # Stochastic oscillator strategy
└── strategies_mine/           # Additional strategy experiments
```

## Modular Framework
ModularFramework
Price
Price Tick
Contract Multiplier
Minimum Tick Value
Minimum Tick Size
Unit Value
Daily Price Volatility
Local Volatility
FX Rate
Base Currency Volatility
Daily Risk Budget
Position Scalar
Aggregate Signal
Raw Position
Allocation Weights
Diversification Factor
Sized Position
Rounded Position
Open Position
Required Trade Size
<img width="170" height="421" alt="image" src="https://github.com/user-attachments/assets/47cfc1c1-5027-49d1-8f36-5b901fafb7d1" />


## Setup Instructions

### Prerequisites
- Python 3.11 or higher
- Conda (recommended) or pip

### 1. Create a Conda Virtual Environment
```bash
conda create --name trading_env python=3.11 -y
```

### 2. Activate the Environment
```bash
conda activate trading_env
```

### 3. Install Required Dependencies
```bash
pip install streamlit pandas numpy matplotlib seaborn scipy h5py openpyxl
```

### 4. Prepare Input Data
Place your instrument CSV files in the `DATA/input_instruments/` folder. Each CSV should contain market data with appropriate columns (price, volume, rates, etc.).

## How to Run the Application

1. **Navigate to the project directory:**
   ```bash
   cd trading_webapp
   ```

2. **Activate the Conda environment:**
   ```bash
   conda activate trading_env
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

4. **Access the application:**
   Open your browser and navigate to `http://localhost:8501`

## Application Workflow

The application follows a sequential workflow across five main pages:

### 1. **Main Analysis Page**
- Reads instrument data from `DATA/input_instruments/`
- Executes multiple trading strategies (EWMA, Carry, Breakout, Stochastic)
- Filters strategies based on cost-effectiveness
- Saves strategy outputs to `DATA/output_instruments/`
- Saves control parameters to `control_output.json`

### 2. **Validation Page**
- Validates strategy outputs and data quality
- Calculates correlation matrices between strategy forecasts
- Identifies highly correlated strategies for diversification analysis
- Generates validation reports

### 3. **PDM (Portfolio Diversification Multiplier) Page**
- Calculates diversification benefits across instruments
- Computes instrument weights based on correlation structure
- Generates PDM metrics for position sizing
- Outputs results to `DATA/combinedForecast/`

### 4. **Forecast Page**
- Combines forecasts from multiple strategies
- Applies IDM (Instrument Diversification Multiplier) adjustments
- Generates position sizes and trading signals
- Creates final forecast files in `DATA/combinedForecast/`

### 5. **Sharpe Ratio Page**
- Calculates risk-adjusted returns for each strategy
- Evaluates historical performance metrics
- Saves Sharpe ratio results to `sharpe_results.json`

## Trading Strategies

### 1. EWMA (Exponentially Weighted Moving Average)
- **Parameters**: Fast/Slow periods (2/8, 4/16, 8/32, 16/64)
- **Signal**: Crossover of fast and slow EMAs
- **Output**: `{Instrument}_ewma_{params}.csv`

### 2. Carry Strategy
- **Types**: 
  - Commodity Carry: Based on futures curve (contango/backwardation)
  - FX Carry: Based on interest rate differentials
- **Signal**: Positive/negative carry opportunities
- **Output**: `{Instrument}_CARRY.csv`

### 3. Breakout Strategy
- **Parameters**: Volatility multiplier and lookback period
- **Signal**: Price breaks above/below volatility bands
- **Output**: `{Instrument}_breakout_{params}.csv`

### 4. Stochastic Strategy
- **Signal**: Oversold/overbought conditions
- **Output**: `{Instrument}_stochastic.csv`

## Data Flow

```
Input CSVs (DATA/input_instruments/)
    ↓
Main Analysis → Strategy Execution
    ↓
Strategy Outputs (DATA/output_instruments/)
    ↓
Validation → Correlation Analysis
    ↓
PDM Calculation → Diversification Metrics
    ↓
Forecast Combination → Position Sizing
    ↓
Order Files (DATA/order_folder/)
```

### Key Output Files

- **`control_output.json`**: Strategy parameters and instrument settings
- **`sharpe_results.json`**: Risk-adjusted performance metrics
- **`{Instrument}_COMBINED.csv`**: Combined forecast with all strategy signals
- **`portfolio_output.csv`**: Final portfolio positions and orders

## Future Enhancements

- **Interactive Brokers Integration**: Direct API connection for automated order placement
- **Real-time Data Feeds**: Live market data integration
- **Backtesting Framework**: Historical performance testing across different time periods
- **Machine Learning Models**: ML-based signal generation and optimization
- **Risk Management Dashboard**: Real-time P&L tracking and risk metrics
- **User Authentication**: Multi-user support with portfolio tracking
- **Advanced Visualization**: Interactive charts with plotly/dash
- **Database Integration**: PostgreSQL/MongoDB for historical data storage
- **Alert System**: Email/SMS notifications for trading signals

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Ensure all packages are installed with `pip install -r requirements.txt` (if available)
2. **Data Format Errors**: Verify input CSV files have required columns (Date, Close, Open, High, Low)
3. **Empty Output Folders**: Check that strategies pass the cost filter (max_payable > standard_cost)
4. **Port Already in Use**: Use `streamlit run app.py --server.port 8502` to use a different port

## Contributing

This is a private trading application. For questions or improvements, contact the development team.

## License

Proprietary - All rights reserved ( Lorenzo ).
