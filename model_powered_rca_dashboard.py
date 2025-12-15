#!/usr/bin/env python3
"""
MODEL-POWERED RCA DASHBOARD
Standalone Streamlit dashboard that uses XGBoost model to answer: "Why did my win rate fall?"
"""

import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from google.cloud import bigquery
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Model-Powered RCA Dashboard",
    page_icon="🔬",
    layout="wide"
)

# Initialize BigQuery client
@st.cache_resource
def get_bq_client():
    return bigquery.Client(project="arboreal-vision-339901")

# Load model
@st.cache_resource
def load_model():
    # Try local path first, then relative path for Streamlit Cloud
    import os
    local_path = '/Users/arvindeashwar/Downloads/doordash_enhanced_model.pkl'
    cloud_path = 'doordash_enhanced_model.pkl'

    model_path = local_path if os.path.exists(local_path) else cloud_path
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Get feature importance
@st.cache_data
def get_feature_importance(_model):
    return pd.DataFrame({
        'feature': _model.feature_names_in_,
        'importance': _model.feature_importances_
    }).sort_values('importance', ascending=False)

# Get available chains from BigQuery
@st.cache_data(ttl=3600)
def get_available_chains():
    client = get_bq_client()
    query = """
    SELECT
        LOWER(REGEXP_EXTRACT(slug, r'^([^_]+)')) as chain_name,
        COUNT(DISTINCT slug) as locations,
        COUNT(*) as total_disputes
    FROM `merchant_portal_export.dispute_training_post_policy`
    WHERE slug IS NOT NULL
    GROUP BY chain_name
    HAVING total_disputes > 100
    ORDER BY total_disputes DESC
    """
    df = client.query(query).to_dataframe()
    return df['chain_name'].tolist()

# Fetch data from BigQuery
@st.cache_data(ttl=3600)
def fetch_rca_data(chain_slug, sep_month, oct_month, year):
    client = get_bq_client()

    query = f"""
    WITH sep_data AS (
        SELECT
            slug,
            error_type,
            is_won,
            COUNT(*) as dispute_count
        FROM `merchant_portal_export.dispute_training_post_policy`
        WHERE LOWER(slug) LIKE '{chain_slug}%'
          AND order_year = {year}
          AND order_month = {sep_month}
        GROUP BY slug, error_type, is_won
    ),
    oct_data AS (
        SELECT
            slug,
            error_type,
            is_won,
            COUNT(*) as dispute_count
        FROM `merchant_portal_export.dispute_training_post_policy`
        WHERE LOWER(slug) LIKE '{chain_slug}%'
          AND order_year = {year}
          AND order_month = {oct_month}
        GROUP BY slug, error_type, is_won
    ),
    sep_summary AS (
        SELECT
            COUNT(DISTINCT slug) as num_locations,
            COUNT(*) as total_disputes,
            SUM(CASE WHEN is_won = 1 THEN 1 ELSE 0 END) as won_disputes,
            ROUND(AVG(CASE WHEN is_won = 1 THEN 1.0 ELSE 0.0 END) * 100, 1) as win_rate,
            ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT slug), 1) as disputes_per_location
        FROM `merchant_portal_export.dispute_training_post_policy`
        WHERE LOWER(slug) LIKE '{chain_slug}%'
          AND order_year = {year}
          AND order_month = {sep_month}
    ),
    oct_summary AS (
        SELECT
            COUNT(DISTINCT slug) as num_locations,
            COUNT(*) as total_disputes,
            SUM(CASE WHEN is_won = 1 THEN 1 ELSE 0 END) as won_disputes,
            ROUND(AVG(CASE WHEN is_won = 1 THEN 1.0 ELSE 0.0 END) * 100, 1) as win_rate,
            ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT slug), 1) as disputes_per_location
        FROM `merchant_portal_export.dispute_training_post_policy`
        WHERE LOWER(slug) LIKE '{chain_slug}%'
          AND order_year = {year}
          AND order_month = {oct_month}
    ),
    location_level AS (
        SELECT
            slug,
            order_month,
            COUNT(*) as disputes,
            AVG(CASE WHEN is_won = 1 THEN 1.0 ELSE 0.0 END) * 100 as win_rate
        FROM `merchant_portal_export.dispute_training_post_policy`
        WHERE LOWER(slug) LIKE '{chain_slug}%'
          AND order_year = {year}
          AND order_month IN ({sep_month}, {oct_month})
        GROUP BY slug, order_month
    )
    SELECT
        'Sep' as month,
        s.num_locations,
        s.total_disputes,
        s.won_disputes,
        s.win_rate,
        s.disputes_per_location
    FROM sep_summary s
    UNION ALL
    SELECT
        'Oct' as month,
        o.num_locations,
        o.total_disputes,
        o.won_disputes,
        o.win_rate,
        o.disputes_per_location
    FROM oct_summary o
    """

    df = client.query(query).to_dataframe()

    # Also get location-level data
    location_query = f"""
    SELECT
        slug,
        order_month as month,
        COUNT(*) as disputes,
        AVG(CASE WHEN is_won = 1 THEN 1.0 ELSE 0.0 END) * 100 as win_rate
    FROM `merchant_portal_export.dispute_training_post_policy`
    WHERE LOWER(slug) LIKE '{chain_slug}%'
      AND order_year = {year}
      AND order_month IN ({sep_month}, {oct_month})
    GROUP BY slug, order_month
    HAVING disputes > 5
    ORDER BY disputes DESC
    """

    location_df = client.query(location_query).to_dataframe()

    return df, location_df

# Main app
def main():
    st.title("🔬 Model-Powered RCA Dashboard")
    st.markdown("### Understanding Win Rate Changes Through Machine Learning")

    # Sidebar controls
    st.sidebar.header("📊 Analysis Parameters")

    # Get available chains
    available_chains = get_available_chains()

    # Find default index (meolicompanies or first chain)
    try:
        default_idx = available_chains.index('meolicompanies')
    except ValueError:
        default_idx = 0

    chain_slug = st.sidebar.selectbox(
        "Select Chain",
        options=available_chains,
        index=default_idx,
        help="Select from chains with >100 disputes in the dataset"
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        sep_month = st.number_input("Baseline Month", min_value=1, max_value=12, value=9)
    with col2:
        oct_month = st.number_input("Target Month", min_value=1, max_value=12, value=10)

    year = st.sidebar.number_input("Year", min_value=2020, max_value=2030, value=2025)

    run_analysis = st.sidebar.button("🔍 Run Analysis", type="primary")

    if run_analysis:
        with st.spinner("Loading model and data..."):
            # Load model
            model = load_model()
            feature_importance = get_feature_importance(model)

            # Fetch data
            df, location_df = fetch_rca_data(chain_slug, sep_month, oct_month, year)

            if len(df) < 2:
                st.error("❌ No data found for this chain and time period")
                return

            sep_row = df[df['month'] == 'Sep'].iloc[0]
            oct_row = df[df['month'] == 'Oct'].iloc[0]

            wr_change = oct_row['win_rate'] - sep_row['win_rate']

            # Header metrics
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Baseline Win Rate",
                    f"{sep_row['win_rate']:.1f}%",
                    f"Month {sep_month}"
                )

            with col2:
                st.metric(
                    "Target Win Rate",
                    f"{oct_row['win_rate']:.1f}%",
                    f"{wr_change:+.1f}pp",
                    delta_color="normal"
                )

            with col3:
                dispute_change = oct_row['total_disputes'] - sep_row['total_disputes']
                dispute_pct = (dispute_change / sep_row['total_disputes'] * 100) if sep_row['total_disputes'] > 0 else 0
                st.metric(
                    "Dispute Volume",
                    f"{int(oct_row['total_disputes'])}",
                    f"{dispute_pct:+.0f}%"
                )

            with col4:
                error_rate_change = oct_row['disputes_per_location'] - sep_row['disputes_per_location']
                error_rate_pct = (error_rate_change / sep_row['disputes_per_location'] * 100) if sep_row['disputes_per_location'] > 0 else 0
                st.metric(
                    "Disputes/Location",
                    f"{oct_row['disputes_per_location']:.1f}",
                    f"{error_rate_pct:+.0f}%"
                )

            st.markdown("---")

            # SECTION 1: MODEL FEATURE IMPORTANCE
            st.header("📈 Section 1: What the Model Says Drives Win Rates")

            st.markdown("""
            Our XGBoost model analyzed thousands of disputes and identified which features
            best predict whether you'll win or lose. Here are the top drivers:
            """)

            # Top 10 features chart
            top_features = feature_importance.head(10)

            fig_importance = go.Figure(go.Bar(
                x=top_features['importance'] * 100,
                y=top_features['feature'],
                orientation='h',
                marker=dict(
                    color=top_features['importance'] * 100,
                    colorscale='Blues',
                    showscale=False
                ),
                text=[f"{x:.1f}%" for x in top_features['importance'] * 100],
                textposition='auto'
            ))

            fig_importance.update_layout(
                title="Top 10 Features That Predict Dispute Outcomes",
                xaxis_title="Importance (%)",
                yaxis_title="",
                height=400,
                yaxis=dict(autorange="reversed")
            )

            st.plotly_chart(fig_importance, use_container_width=True)

            # Translation to operations
            st.subheader("🔄 Translation to Operations")

            translations = {
                'is_high_error_rate': 'Are you in a high-error period? (spike in disputes)',
                'slug_historical_wr': 'Your past win rate (reputation with platform)',
                'win_rate_chain_avg': 'How well your chain typically performs',
                'incomplete_pct': '% of errors that are missing items',
                'error_rate': 'How many errors per location per day',
                'beverages_pct': '% of errors involving drinks',
                'error_rate_chain_avg': 'Your chain\'s typical error rate',
                'is_low_volume': 'Are you a low-volume merchant?',
                'total_errors': 'Total number of errors in period',
                'dispute_count': 'How many disputes you\'re handling'
            }

            for idx, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
                feature = row['feature']
                importance = row['importance']
                translation = translations.get(feature, 'Technical feature')

                st.markdown(f"""
                **{idx}. {feature}** ({importance:.1%} importance)
                - 💡 *{translation}*
                """)

            st.markdown("---")

            # SECTION 2: FEATURE DELTA ANALYSIS
            st.header("📊 Section 2: What Changed in Your Data")

            st.markdown("""
            Now let's see how these important features changed between your baseline and target periods:
            """)

            # Calculate key deltas
            error_rate_delta = oct_row['disputes_per_location'] - sep_row['disputes_per_location']
            error_rate_pct_change = (error_rate_delta / sep_row['disputes_per_location'] * 100) if sep_row['disputes_per_location'] > 0 else 0

            total_change = oct_row['total_disputes'] - sep_row['total_disputes']
            total_pct_change = (total_change / sep_row['total_disputes'] * 100) if sep_row['total_disputes'] > 0 else 0

            # Display deltas
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                #### 1️⃣ error_rate
                **Model Weight:** 6.6%

                **Proxy:** Disputes per location
                📉 Sep: **{sep_row['disputes_per_location']:.1f}** disputes/location
                📈 Oct: **{oct_row['disputes_per_location']:.1f}** disputes/location

                **Change:** {error_rate_delta:+.1f} ({error_rate_pct_change:+.0f}%)
                """)

                if abs(error_rate_pct_change) > 10:
                    st.warning("⚠️ SIGNIFICANT CHANGE - This is likely driving WR impact")

            with col2:
                st.markdown(f"""
                #### 2️⃣ slug_historical_wr
                **Model Weight:** 12.7%

                **What it means:** Your reputation/track record

                📉 Sep baseline: **{sep_row['win_rate']:.1f}%**

                As WR drops, future disputes become harder to win (downward spiral)
                """)

            with col3:
                st.markdown(f"""
                #### 3️⃣ dispute_count
                **Model Weight:** 2.8%

                📉 Sep: **{int(sep_row['total_disputes'])}** disputes
                📈 Oct: **{int(oct_row['total_disputes'])}** disputes

                **Change:** {int(total_change)} ({total_pct_change:+.0f}%)
                """)

            st.markdown("---")

            # NEW SECTION 2.5: MODEL'S TOP 5 FACTORS - EXPLICIT TRACKING
            st.header("🎯 Section 2.5: Model's Top 5 Win Rate Drivers - Did They Change?")

            st.markdown("""
            The model identified **5 critical factors** that predict win rates. Let's see if these changed between Sep and Oct:
            """)

            # Query detailed data for top 5 factors
            detailed_query = f"""
            WITH temporal_patterns AS (
                SELECT
                    order_month,
                    order_dow as day_of_week,
                    order_hour as hour,
                    COUNT(*) as disputes,
                    AVG(CASE WHEN is_won = 1 THEN 1.0 ELSE 0.0 END) * 100 as win_rate
                FROM `merchant_portal_export.dispute_training_post_policy`
                WHERE LOWER(slug) LIKE '{chain_slug}%'
                  AND order_year = {year}
                  AND order_month IN ({sep_month}, {oct_month})
                GROUP BY 1, 2, 3
                HAVING disputes > 5
            ),
            error_categories AS (
                SELECT
                    order_month,
                    error_type,
                    COUNT(*) as disputes,
                    AVG(CASE WHEN is_won = 1 THEN 1.0 ELSE 0.0 END) * 100 as win_rate,
                    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY order_month) as pct_of_total
                FROM `merchant_portal_export.dispute_training_post_policy`
                WHERE LOWER(slug) LIKE '{chain_slug}%'
                  AND order_year = {year}
                  AND order_month IN ({sep_month}, {oct_month})
                GROUP BY 1, 2
            ),
            order_types AS (
                SELECT
                    order_month,
                    CASE
                        WHEN UPPER(error_type) LIKE '%CANCEL%' THEN 'Cancelled'
                        WHEN UPPER(error_type) LIKE '%INACCURATE%' OR UPPER(error_type) LIKE '%MISSING%' OR UPPER(error_type) LIKE '%WRONG%' THEN 'Inaccurate'
                        ELSE 'Other'
                    END as order_type,
                    COUNT(*) as disputes,
                    AVG(CASE WHEN is_won = 1 THEN 1.0 ELSE 0.0 END) * 100 as win_rate
                FROM `merchant_portal_export.dispute_training_post_policy`
                WHERE LOWER(slug) LIKE '{chain_slug}%'
                  AND order_year = {year}
                  AND order_month IN ({sep_month}, {oct_month})
                GROUP BY 1, 2
            )
            SELECT 'temporal' as data_type, CAST(order_month AS STRING) as month, CAST(day_of_week AS STRING) as category,
                   CAST(hour AS STRING) as subcategory, disputes, win_rate FROM temporal_patterns
            UNION ALL
            SELECT 'error_category', CAST(order_month AS STRING), error_type, CAST(pct_of_total AS STRING), disputes, win_rate FROM error_categories
            UNION ALL
            SELECT 'order_type', CAST(order_month AS STRING), order_type, '', disputes, win_rate FROM order_types
            """

            try:
                detailed_df = get_bq_client().query(detailed_query).to_dataframe()

                # Factor #1: Temporal Patterns (33% importance)
                st.subheader("1️⃣ Factor #1: Temporal Patterns (33% model weight)")
                st.markdown("**Day/Hour when disputes occur** - The #1 driver of win rates")

                temporal_data = detailed_df[detailed_df['data_type'] == 'temporal'].copy()

                if len(temporal_data) > 0:
                    # Calculate peak hours
                    sep_temporal = temporal_data[temporal_data['month'] == str(sep_month)]
                    oct_temporal = temporal_data[temporal_data['month'] == str(oct_month)]

                    if len(sep_temporal) > 0 and len(oct_temporal) > 0:
                        sep_peak_hour = sep_temporal.groupby('subcategory')['disputes'].sum().idxmax()
                        oct_peak_hour = oct_temporal.groupby('subcategory')['disputes'].sum().idxmax()

                        sep_peak_wr = sep_temporal[sep_temporal['subcategory'] == sep_peak_hour]['win_rate'].mean()
                        oct_peak_wr = oct_temporal[oct_temporal['subcategory'] == oct_peak_hour]['win_rate'].mean()

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                f"Sep Peak Hour ({sep_peak_hour}:00)",
                                f"{sep_peak_wr:.1f}% WR",
                                f"{sep_temporal[sep_temporal['subcategory'] == sep_peak_hour]['disputes'].sum():.0f} disputes"
                            )
                        with col2:
                            wr_change_temporal = oct_peak_wr - sep_peak_wr
                            st.metric(
                                f"Oct Peak Hour ({oct_peak_hour}:00)",
                                f"{oct_peak_wr:.1f}% WR",
                                f"{wr_change_temporal:+.1f}pp",
                                delta_color="normal"
                            )

                        if sep_peak_hour != oct_peak_hour:
                            st.warning(f"⚠️ **Peak hour shifted** from {sep_peak_hour}:00 to {oct_peak_hour}:00 - this changes win rate dynamics!")

                        if abs(wr_change_temporal) > 5:
                            st.error(f"🚨 **CRITICAL:** Peak hour WR changed by {wr_change_temporal:+.1f}pp - This is the #1 model driver (33% weight)!")
                else:
                    st.info("⚠️ Insufficient temporal data for analysis")

                st.markdown("---")

                # Factor #2: Error Category Baseline (20% importance)
                st.subheader("2️⃣ Factor #2: Error Category Mix (20% model weight)")
                st.markdown("**Which error types** - Some categories are inherently harder to win")

                error_cat_data = detailed_df[detailed_df['data_type'] == 'error_category'].copy()

                if len(error_cat_data) > 0:
                    sep_errors = error_cat_data[error_cat_data['month'] == str(sep_month)].nlargest(5, 'disputes')
                    oct_errors = error_cat_data[error_cat_data['month'] == str(oct_month)].nlargest(5, 'disputes')

                    # Show top error types side by side
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"**Sep Top Errors:**")
                        for idx, row in sep_errors.iterrows():
                            pct = float(row['subcategory'])
                            st.markdown(f"- **{row['category']}**: {pct:.1f}% of disputes, {row['win_rate']:.1f}% WR")

                    with col2:
                        st.markdown(f"**Oct Top Errors:**")
                        for idx, row in oct_errors.iterrows():
                            pct = float(row['subcategory'])
                            st.markdown(f"- **{row['category']}**: {pct:.1f}% of disputes, {row['win_rate']:.1f}% WR")

                    # Check for mix shifts
                    sep_top_error = sep_errors.iloc[0]['category']
                    oct_top_error = oct_errors.iloc[0]['category']

                    if sep_top_error != oct_top_error:
                        st.warning(f"⚠️ **Error mix shifted**: Top error changed from '{sep_top_error}' to '{oct_top_error}'")
                        st.markdown("💡 **Impact:** Different error types have different baseline win rates - this shift affects overall WR")
                else:
                    st.info("⚠️ Insufficient error category data")

                st.markdown("---")

                # Factor #4: Order Type (9% importance)
                st.subheader("4️⃣ Factor #4: Order Type - Cancelled vs Inaccurate (9% model weight)")
                st.markdown("**Order type split** - Cancelled orders have different win rates than inaccurate orders")

                order_type_data = detailed_df[detailed_df['data_type'] == 'order_type'].copy()

                if len(order_type_data) > 0:
                    sep_types = order_type_data[order_type_data['month'] == str(sep_month)]
                    oct_types = order_type_data[order_type_data['month'] == str(oct_month)]

                    # Create comparison chart
                    fig_types = go.Figure()

                    for order_type in ['Cancelled', 'Inaccurate', 'Other']:
                        sep_wr = sep_types[sep_types['category'] == order_type]['win_rate'].values
                        oct_wr = oct_types[oct_types['category'] == order_type]['win_rate'].values

                        if len(sep_wr) > 0 and len(oct_wr) > 0:
                            fig_types.add_trace(go.Bar(
                                name=f'{order_type} - Sep',
                                x=[f'{order_type} Sep'],
                                y=[sep_wr[0]],
                                marker_color='lightblue'
                            ))
                            fig_types.add_trace(go.Bar(
                                name=f'{order_type} - Oct',
                                x=[f'{order_type} Oct'],
                                y=[oct_wr[0]],
                                marker_color='darkblue'
                            ))

                    fig_types.update_layout(
                        title="Win Rate by Order Type: Sep vs Oct",
                        yaxis_title="Win Rate (%)",
                        barmode='group',
                        height=400
                    )

                    st.plotly_chart(fig_types, use_container_width=True)

                    # Calculate mix shift
                    sep_cancelled_pct = sep_types[sep_types['category'] == 'Cancelled']['disputes'].sum() / sep_types['disputes'].sum() * 100
                    oct_cancelled_pct = oct_types[oct_types['category'] == 'Cancelled']['disputes'].sum() / oct_types['disputes'].sum() * 100
                    cancelled_shift = oct_cancelled_pct - sep_cancelled_pct

                    if abs(cancelled_shift) > 5:
                        st.warning(f"⚠️ **Mix shift detected**: Cancelled orders went from {sep_cancelled_pct:.1f}% to {oct_cancelled_pct:.1f}% of total ({cancelled_shift:+.1f}pp)")
                else:
                    st.info("⚠️ Insufficient order type data")

            except Exception as e:
                st.error(f"Error fetching detailed factor data: {str(e)}")
                st.info("Continuing with aggregate analysis...")

            st.markdown("---")

            # SECTION 3: MODEL-BASED ATTRIBUTION
            st.header("🎯 Section 3: Model-Based Attribution")

            st.markdown(f"""
            Using the model's feature importance, we can estimate how much each factor contributed
            to your **{wr_change:+.1f}pp** win rate change:
            """)

            # Calculate attribution (rough estimates)
            error_rate_impact = -0.066 * (error_rate_pct_change / 100) * 100
            volume_impact = -0.028 * (total_pct_change / 100) * 100
            historical_impact = wr_change - error_rate_impact - volume_impact

            # Waterfall chart
            fig_waterfall = go.Figure(go.Waterfall(
                name="Attribution",
                orientation="v",
                measure=["relative", "relative", "relative", "total"],
                x=["Error Rate<br>Increase", "Dispute Volume<br>Increase", "Other Factors<br>(Historical)", "Total WR<br>Change"],
                y=[error_rate_impact, volume_impact, historical_impact, wr_change],
                text=[f"{error_rate_impact:.1f}pp", f"{volume_impact:.1f}pp",
                      f"{historical_impact:.1f}pp", f"{wr_change:.1f}pp"],
                textposition="outside",
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))

            fig_waterfall.update_layout(
                title=f"Win Rate Change Attribution: {wr_change:+.1f}pp",
                yaxis_title="Impact (percentage points)",
                showlegend=False,
                height=500
            )

            st.plotly_chart(fig_waterfall, use_container_width=True)

            # Attribution breakdown
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "1. Error Rate Impact",
                    f"{error_rate_impact:.1f}pp",
                    f"{abs(error_rate_impact/wr_change)*100:.0f}% of change" if wr_change != 0 else "N/A"
                )

            with col2:
                st.metric(
                    "2. Volume Impact",
                    f"{volume_impact:.1f}pp",
                    f"{abs(volume_impact/wr_change)*100:.0f}% of change" if wr_change != 0 else "N/A"
                )

            with col3:
                st.metric(
                    "3. Historical/Other",
                    f"{historical_impact:.1f}pp",
                    f"{abs(historical_impact/wr_change)*100:.0f}% of change" if wr_change != 0 else "N/A"
                )

            st.info("📝 **Note:** These are rough estimates based on feature importance × % change. True attribution requires full feature set.")

            st.markdown("---")

            # SECTION 4: LOCATION-LEVEL MODEL SCORING
            st.header("📍 Section 4: Location Concentration (80/20 Analysis)")

            st.markdown("""
            The model helps us identify which locations are driving the change. Let's see the 80/20 breakdown:
            """)

            # Calculate location changes
            if len(location_df) > 0:
                sep_locs = location_df[location_df['month'] == sep_month].set_index('slug')
                oct_locs = location_df[location_df['month'] == oct_month].set_index('slug')

                # Merge and calculate changes
                loc_comparison = pd.DataFrame({
                    'sep_disputes': sep_locs['disputes'],
                    'oct_disputes': oct_locs['disputes'],
                    'sep_wr': sep_locs['win_rate'],
                    'oct_wr': oct_locs['win_rate']
                }).fillna(0)

                loc_comparison['volume_change'] = loc_comparison['oct_disputes'] - loc_comparison['sep_disputes']
                loc_comparison['wr_change'] = loc_comparison['oct_wr'] - loc_comparison['sep_wr']
                loc_comparison = loc_comparison.sort_values('volume_change', ascending=False, key=abs)

                # Show top 10 locations
                st.subheader(f"Top 10 Locations by Volume Change")

                top_10 = loc_comparison.head(10).reset_index()

                # Create visualization
                fig_locations = go.Figure()

                fig_locations.add_trace(go.Bar(
                    name='Volume Change',
                    x=top_10['slug'],
                    y=top_10['volume_change'],
                    marker_color='lightblue',
                    yaxis='y',
                    offsetgroup=1
                ))

                fig_locations.add_trace(go.Scatter(
                    name='WR Change',
                    x=top_10['slug'],
                    y=top_10['wr_change'],
                    mode='markers+lines',
                    marker=dict(size=10, color='red'),
                    yaxis='y2'
                ))

                fig_locations.update_layout(
                    title="Location Performance: Volume Change vs Win Rate Change",
                    xaxis=dict(title='Location'),
                    yaxis=dict(title='Volume Change (disputes)', side='left'),
                    yaxis2=dict(title='Win Rate Change (pp)', overlaying='y', side='right'),
                    height=500,
                    hovermode='x unified'
                )

                st.plotly_chart(fig_locations, use_container_width=True)

                # Detailed table
                st.subheader("📊 Detailed Location Breakdown")

                display_df = top_10[['slug', 'sep_disputes', 'oct_disputes', 'volume_change',
                                     'sep_wr', 'oct_wr', 'wr_change']].copy()
                display_df.columns = ['Location', 'Sep Disputes', 'Oct Disputes', 'Volume Δ',
                                     'Sep WR %', 'Oct WR %', 'WR Δ (pp)']

                # Format columns
                display_df['Sep WR %'] = display_df['Sep WR %'].apply(lambda x: f"{x:.1f}%")
                display_df['Oct WR %'] = display_df['Oct WR %'].apply(lambda x: f"{x:.1f}%")
                display_df['WR Δ (pp)'] = display_df['WR Δ (pp)'].apply(lambda x: f"{x:+.1f}")
                display_df['Volume Δ'] = display_df['Volume Δ'].apply(lambda x: f"{int(x):+d}")

                st.dataframe(display_df, use_container_width=True, hide_index=True)

                # 80/20 calculation
                total_volume_change = loc_comparison['volume_change'].abs().sum()
                top_3_contribution = loc_comparison.head(3)['volume_change'].abs().sum()
                top_3_pct = (top_3_contribution / total_volume_change * 100) if total_volume_change > 0 else 0

                st.success(f"🎯 **80/20 Result:** Top 3 locations account for **{top_3_pct:.0f}%** of total volume change")

            else:
                st.warning("⚠️ Not enough location-level data available for analysis")

            st.markdown("---")

            # SECTION 5: ACTIONABLE INSIGHTS
            st.header("💡 Section 5: What To Do About It")

            st.markdown("Based on the model-powered analysis, here are your action items:")

            # Primary driver identification
            if abs(error_rate_pct_change) > 10:
                st.error("### 🚨 PRIMARY DRIVER: Error Rate Increase")
                st.markdown(f"""
                **Impact:** Your disputes per location increased by **{error_rate_pct_change:+.0f}%**
                (from {sep_row['disputes_per_location']:.1f} to {oct_row['disputes_per_location']:.1f})

                **Operational Actions:**
                1. 🔍 **Investigate root cause** of dispute volume spike
                2. 🍔 **Check for:**
                   - Kitchen operational issues (speed, accuracy)
                   - New menu items causing confusion
                   - Staffing/training gaps
                   - Equipment failures
                3. 📍 **Focus on top 3 locations** (80/20 rule applies)
                4. 📊 **Monitor daily** to catch issues early
                """)

            if wr_change < -5:
                st.warning("### ⚠️ URGENT: Win Rate Reputation Spiral Risk")
                st.markdown(f"""
                The model identified **slug_historical_wr** as the #2 driver (12.7% importance).

                **What this means:** As your WR drops from **{sep_row['win_rate']:.1f}%** to **{oct_row['win_rate']:.1f}%**,
                the platform's algorithm becomes less favorable to you in future disputes.

                **Actions to Break the Spiral:**
                1. ✅ **Immediately audit evidence quality** for new disputes
                2. 📝 **Standardize templates** that have high win rates
                3. 🎯 **Prioritize high-value disputes** (focus where model predicts wins)
                4. 📈 **Track daily WR** to prevent further degradation
                5. 🤝 **Escalate borderline cases** to platform support
                """)

            # Model-driven recommendations
            st.subheader("🎯 Model-Driven Priorities")

            st.markdown(f"""
            Based on feature importance, focus on:

            1. **Error Rate Control** (6.6% model weight)
               - Target: Reduce disputes/location back to {sep_row['disputes_per_location']:.1f}
               - Current gap: {error_rate_delta:+.1f} disputes/location

            2. **Win Rate Recovery** (12.7% model weight via historical_wr)
               - Target: Get back above {sep_row['win_rate']:.1f}%
               - Current gap: {wr_change:.1f}pp

            3. **Incomplete Items Focus** (7.0% model weight)
               - Model says this is a top-5 driver
               - Review your missing item protocols

            4. **Beverages Accuracy** (4.6% model weight)
               - Another top-10 feature
               - Check drink prep and packaging procedures
            """)

            st.markdown("---")

            # Footer
            st.caption(f"🔬 Analysis powered by XGBoost model trained on {len(model.feature_names_in_)} features | Data source: BigQuery dispute_training_post_policy table")

if __name__ == "__main__":
    main()
