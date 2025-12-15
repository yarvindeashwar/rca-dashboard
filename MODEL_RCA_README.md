# Model-Powered RCA Dashboard

A production-ready Streamlit dashboard that uses XGBoost machine learning to answer: **"Why did my win rate fall?"**

## What Makes This Special

Unlike traditional analytics dashboards that only show statistics, this dashboard is **truly model-powered**:

- Uses XGBoost model to identify which features predict dispute outcomes
- Calculates feature deltas to show what changed between periods
- Provides model-based attribution showing impact of each factor
- Translates technical ML features into operational business language
- Offers actionable, Loop-aware recommendations

## Key Features

### 1. Model Feature Importance
- Shows top 10 features that predict dispute outcomes
- Interactive visualization with importance percentages
- Translates ML features to business language:
  - `is_high_error_rate` → "Are you in a high-error period?"
  - `slug_historical_wr` → "Your past win rate (reputation)"
  - `error_rate` → "How many errors per location per day"

### 2. Feature Delta Analysis
- Shows how important model features changed between periods
- Auto-highlights significant changes (>10%)
- Three key metrics:
  - error_rate (6.6% model weight)
  - slug_historical_wr (12.7% model weight)
  - dispute_count (2.8% model weight)

### 3. Model-Based Attribution
- Waterfall chart showing impact breakdown
- Uses feature importance × % change for attribution
- Quantifies contribution of each factor to win rate change

### 4. Location-Level Scoring (80/20 Analysis)
- Identifies which locations drive the change
- Dual-axis visualization: Volume Change vs Win Rate Change
- Calculates concentration metrics

### 5. Actionable Insights
- Model-driven recommendations based on feature importance
- Loop-aware (no blame on defense quality)
- Prioritized action items tied to model weights

## Requirements

### Files Needed
```
/Users/arvindeashwar/Downloads/
├── model_powered_rca_dashboard.py  # Main dashboard
└── doordash_enhanced_model.pkl     # XGBoost model (1.1MB)
```

### Python Dependencies
```
streamlit
pandas
plotly
pickle
google-cloud-bigquery
```

### Data Access
- BigQuery project: `arboreal-vision-339901`
- Table: `merchant_portal_export.dispute_training_post_policy`
- Required permissions: BigQuery Data Viewer

## How to Run

### Quick Start
```bash
export GOOGLE_CLOUD_PROJECT="arboreal-vision-339901"
streamlit run model_powered_rca_dashboard.py --server.port 8509
```

### Access
Open browser to: **http://localhost:8509**

## Usage Instructions

### Step 1: Select Chain
- Dropdown shows 20+ chains sorted by data volume
- Includes: aprio, whataburger, sunholdings, daves, freddys, etc.
- Only chains with >100 disputes shown (ensures meaningful analysis)

### Step 2: Choose Time Periods
- **Baseline Month**: Reference period (e.g., 9 for September)
- **Target Month**: Period to analyze (e.g., 10 for October)
- **Year**: Currently supports 2025 (months 5-11 available)

### Step 3: Run Analysis
Click "Run Analysis" button to:
1. Load XGBoost model and extract feature importance
2. Query BigQuery for chain's dispute data
3. Calculate feature deltas between periods
4. Generate model-based attribution
5. Show location-level scoring
6. Provide actionable recommendations

## Dashboard Sections Explained

### Section 1: What the Model Says Drives Win Rates
**Purpose**: Shows which features the ML model uses to predict outcomes

**How to Read**: Higher importance % = stronger predictor
- Top feature (13.4%): `is_high_error_rate` - spike detection
- Second (12.7%): `slug_historical_wr` - your reputation
- Third (9.4%): `win_rate_chain_avg` - chain performance

**Action**: Focus on top 5 features for maximum impact

### Section 2: What Changed in Your Data
**Purpose**: Shows how model features changed between your two periods

**How to Read**:
- Green/red indicators show direction of change
- Percentage change from baseline
- Warning appears if change >10% (significant)

**Action**: Investigate features with large changes

### Section 3: Model-Based Attribution
**Purpose**: Quantifies how much each factor contributed to WR change

**How to Read**:
- Waterfall chart shows cumulative impact
- Formula: Impact = Feature Importance × % Change × 100
- Adds up to total WR change

**Action**: Prioritize factors with largest negative impact

### Section 4: Location Concentration (80/20)
**Purpose**: Identifies which locations drive the change

**How to Read**:
- Sorted by volume change (largest first)
- Shows both volume and WR change
- Calculates % of total change from top locations

**Action**: Focus operational efforts on top 3 locations

### Section 5: What To Do About It
**Purpose**: Provides model-driven, actionable recommendations

**How to Read**:
- Conditional logic based on model insights
- Prioritized by feature importance
- Loop-aware (focuses on inputs, not defense)

**Action**: Follow prioritized recommendations

## Model Details

### XGBoost Model Specifications
- **Features**: 31 total
- **Training Data**: DoorDash post-policy disputes
- **Target**: Binary win/loss outcome
- **File**: `doordash_enhanced_model.pkl` (1.1MB)

### Top 10 Features by Importance
1. is_high_error_rate (13.4%)
2. slug_historical_wr (12.7%)
3. win_rate_chain_avg (9.4%)
4. incomplete_pct (7.0%)
5. error_rate (6.6%)
6. beverages_pct (4.6%)
7. error_rate_chain_avg (4.5%)
8. is_low_volume (3.8%)
9. total_errors (2.8%)
10. dispute_count (2.8%)

### Attribution Formula
```
Impact = Feature_Importance × (% Change / 100) × 100

Total WR Change = Σ Individual Impacts
```

Example:
- error_rate increased by 383%
- Feature importance: 6.6%
- Impact: -0.066 × (383/100) × 100 = -25.3pp

## Data Schema

### Source Table
```sql
merchant_portal_export.dispute_training_post_policy

Columns:
- slug: Location identifier
- order_year: Year of order
- order_month: Month of order
- is_won: Win (1) or Loss (0)
- error_type: Type of error (INACCURATE, INCOMPLETE, etc.)
```

### Chain Extraction
```sql
LOWER(REGEXP_EXTRACT(slug, r'^([^_]+)'))
```

Example: `doghaus_bk_2477385_dd` → `doghaus`

## Troubleshooting

### "No data found for this chain and time period"
- Check if chain has disputes in those months
- Try different month combinations
- Verify chain slug spelling

### "Model file not found"
- Ensure `doordash_enhanced_model.pkl` is in same directory
- Check file path in code (line 30)

### BigQuery Permission Error
- Verify Google Cloud credentials are set
- Check project ID: `arboreal-vision-339901`
- Ensure service account has BigQuery Data Viewer role

### Slow Performance
- Data is cached for 1 hour (TTL=3600)
- First load may be slow (querying BigQuery)
- Subsequent loads use cache

## Best Practices

### For Analysts
1. Always compare consecutive months first
2. Look for >10% changes in key features
3. Cross-reference model insights with operational knowledge
4. Focus on top 3 locations (80/20 rule)

### For Operations Teams
1. Start with Section 5 (Actionable Insights)
2. Use Section 4 to identify problem locations
3. Section 2 shows which metrics to track
4. Ignore technical ML jargon - focus on translations

### For Executives
1. Read header metrics only (4 cards at top)
2. Review Section 3 waterfall chart
3. Check Section 5 for strategic actions
4. Drill into details only if needed

## File Location

**Dashboard**: `/Users/arvindeashwar/Downloads/model_powered_rca_dashboard.py`

**Model**: `/Users/arvindeashwar/Downloads/doordash_enhanced_model.pkl`

**URL**: http://localhost:8509

## Support

For questions or issues:
1. Check this README first
2. Review error messages in Streamlit
3. Verify BigQuery access and credentials
4. Check model file exists and is readable

## Version History

**v1.0** (Current)
- Initial production release
- Supports 20+ chains
- All months (5-11) in 2025
- 5 analysis sections
- Model-powered attribution
- Location-level scoring
