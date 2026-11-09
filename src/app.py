"""
app.py — Customer Segmentation Streamlit Dashboard
Author: Ashutosh Behera | Data Scientist | Bandhan Bank Ltd
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import boto3
import io
import os
import time
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from segment import assign_segment, assign_batch, load_model_artifacts, SEGMENT_STRATEGIES
from llm_messages import generate_message, generate_batch_messages

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Big Basket — Customer Segmentation",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.segment-card {
    padding: 20px; border-radius: 12px;
    margin: 12px 0; border-left: 6px solid;
}
.chat-bubble {
    background: #f0f2f6; border-radius: 18px 18px 18px 4px;
    padding: 16px 20px; margin: 10px 0;
    border-left: 4px solid #1F4E79;
    font-size: 15px; line-height: 1.6;
}
.chat-bubble-header {
    font-size: 11px; color: #888; margin-bottom: 8px;
}
.metric-card {
    background: white; border-radius: 10px;
    padding: 16px; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin: 6px;
}
.urgency-critical { color: #7030A0; font-weight: bold; }
.urgency-high     { color: #C00000; font-weight: bold; }
.urgency-medium   { color: #C55A11; font-weight: bold; }
.urgency-low      { color: #2e7d32; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

COLORS = {
    "Champions":           "#1F4E79",
    "Loyal Customers":     "#2e7d32",
    "Potential Loyalists": "#C55A11",
    "At Risk":             "#C00000",
    "Cannot Lose":         "#7030A0",
    "Hibernating":         "#833C00",
    "Lost Customers":      "#4472C4",
}
URGENCY_ICON = {"CRITICAL":"🚨","HIGH":"⚠️","medium":"📋","low":"📬"}

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Big Basket")
    st.markdown("### Customer Segmentation")
    st.markdown("**Author:** Ashutosh Behera | Bandhan Bank Ltd")
    st.markdown("---")

    try:
        artifacts = load_model_artifacts()
        st.success(f"✅ Model Ready\n{len(artifacts['segment_names'])} segments loaded")
    except Exception as e:
        st.error(f"❌ {e}")

    st.markdown("---")
    st.markdown("### 📊 Segment Guide")
    for seg, color in COLORS.items():
        strat  = SEGMENT_STRATEGIES.get(seg, {})
        urgency = strat.get("urgency","low")
        icon   = URGENCY_ICON.get(urgency,"📬")
        st.markdown(
            f"<div style='padding:4px 0'>"
            f"<span style='color:{color};font-size:18px'>●</span> "
            f"<strong>{seg}</strong> {icon}</div>",
            unsafe_allow_html=True
        )
    st.markdown("---")
    st.caption("Powered by K-Means++ Centroid Mapping + OpenAI GPT-4o-mini")

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🛒 Customer Segmentation — Big Basket")
st.markdown("**Assign customers to segments instantly. Generate personalised messages. Drive revenue.**")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Single Customer Lookup",
    "📂 Batch CSV Upload",
    "📈 Segment Overview Dashboard"
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE CUSTOMER
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Predict Segment for One Customer")
    st.markdown("Enter customer details → get instant segment + personalised WhatsApp/Email message")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 👤 Customer Profile")
        name           = st.text_input("Full Name", "Priya Sharma")
        customer_id    = st.text_input("Customer ID", "CUST200001")
        email          = st.text_input("Email", "priya.sharma@email.com")
        phone          = st.text_input("Phone", "+91 9876543210")
        gender         = st.selectbox("Gender", ["Female","Male","Other"])
        income_segment = st.selectbox("Income Segment", ["Low","Mid","High","Ultra"], index=2)
        city_tier      = st.selectbox("City Tier", ["Metro","Tier2","Tier3"])
        city           = st.text_input("City", "Mumbai")

    with col2:
        st.markdown("#### 💰 Purchase Behaviour")
        recency_days     = st.number_input("Recency Days (since last purchase)", 1, 365, 45)
        frequency        = st.number_input("Frequency (orders/year)", 1, 100, 24)
        monetary_value   = st.number_input("Annual Spend (Rs)", 500, 200000, 22000, step=500)
        avg_order_value  = st.number_input("Avg Order Value (Rs)", 100, 15000,
                                            max(200, int(monetary_value/max(frequency,1))))
        tenure_months    = st.number_input("Tenure (months)", 1, 120, 18)
        discount_rate    = st.slider("Discount Usage Rate", 0.0, 1.0, 0.30, step=0.01)
        purchase_gap_trend = st.slider("Purchase Gap Trend (positive = widening)", -0.5, 1.0, 0.10, step=0.01)
        favourite_category = st.selectbox("Favourite Category", [
            "Fruits & Vegetables","Dairy & Eggs","Snacks & Beverages",
            "Staples & Grains","Personal Care","Cleaning & Household",
            "Meat & Seafood","Organic & Gourmet","Baby & Kids","Frozen Foods"])

    with col3:
        st.markdown("#### 📱 Engagement Signals")
        loyalty_enrolled      = st.checkbox("Loyalty Programme Member", True)
        app_user              = st.checkbox("Mobile App User", True)
        response_to_campaign  = st.checkbox("Responded to Last Campaign", False)
        support_tickets       = st.number_input("Support Tickets Filed", 0, 20, 1)
        premium_brand_ratio   = st.slider("Premium Brand Preference", 0.0, 1.0, 0.30)
        organic_ratio         = st.slider("Organic Product Preference", 0.0, 1.0, 0.20)
        has_children          = st.checkbox("Has Children", False)

        st.markdown("---")
        st.markdown("#### 🤖 LLM Options")
        channel  = st.radio("Campaign Channel", ["WhatsApp","Email","SMS"], horizontal=True)
        use_llm  = st.checkbox("Generate AI Message (requires OpenAI key)", True)
        openai_key = st.text_input("OpenAI API Key (optional)", type="password",
                                    placeholder="sk-...")

    st.markdown("---")
    predict_btn = st.button("🎯 Predict Segment & Generate Message",
                            type="primary", use_container_width=True)

    if predict_btn:
        r_sc = 5 if recency_days<=7 else 4 if recency_days<=20 else 3 if recency_days<=50 else 2 if recency_days<=120 else 1
        f_sc = 5 if frequency>=48 else 4 if frequency>=24 else 3 if frequency>=12 else 2 if frequency>=4 else 1
        m_sc = 5 if monetary_value>=50000 else 4 if monetary_value>=25000 else 3 if monetary_value>=10000 else 2 if monetary_value>=3000 else 1

        record = {
            "customer_id": customer_id, "name": name, "email": email,
            "recency_days": recency_days, "frequency": frequency,
            "monetary_value": monetary_value, "avg_order_value": avg_order_value,
            "tenure_months": tenure_months, "discount_usage_rate": discount_rate,
            "purchase_gap_trend": purchase_gap_trend,
            "loyalty_enrolled": int(loyalty_enrolled), "app_user": int(app_user),
            "response_to_campaign": int(response_to_campaign),
            "support_tickets": support_tickets, "has_children": int(has_children),
            "premium_brand_ratio": premium_brand_ratio,
            "organic_product_ratio": organic_ratio,
            "income_segment": income_segment, "city_tier": city_tier,
            "city": city, "gender": gender,
            "favourite_category": favourite_category,
            "recency_score": r_sc, "frequency_score": f_sc,
            "monetary_score": m_sc, "rfm_score": r_sc+f_sc+m_sc,
        }

        with st.spinner("Assigning segment using centroid mapping..."):
            result = assign_segment(record)

        seg    = result["segment_name"]
        conf   = result["confidence"]
        color  = COLORS.get(seg, "#333")
        strat  = result["strategy"]
        urgency_icon = URGENCY_ICON.get(strat["urgency"], "📬")

        # ── Segment Result Card ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🎯 Segment Result")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Segment", seg)
        c2.metric("Confidence", f"{conf:.1%}")
        c3.metric("Urgency", strat["urgency"])
        c4.metric("Best Channel", strat["channel"])

        st.markdown(f"""
<div class="segment-card" style="border-color:{color}; background:{color}18;">
  <h3 style="color:{color}; margin:0 0 12px 0">{urgency_icon} {seg}</h3>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
    <div>
      <p style="margin:4px 0"><strong>📋 Description:</strong> {strat['description']}</p>
      <p style="margin:4px 0"><strong>🎁 Recommended Offer:</strong> {strat['offer']}</p>
    </div>
    <div>
      <p style="margin:4px 0"><strong>📡 Channel:</strong> {strat['channel']}</p>
      <p style="margin:4px 0"><strong>🎤 Tone:</strong> {strat['tone']}</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        # ── Centroid Distance Bar ───────────────────────────────────────
        all_dists = result.get("all_distances", {})
        if all_dists:
            dist_df = pd.DataFrame(list(all_dists.items()),
                                   columns=["Segment","Distance"]).sort_values("Distance")
            fig, ax = plt.subplots(figsize=(10, 3.5))
            bar_colors = [COLORS.get(s,"#ccc") for s in dist_df["Segment"]]
            bars = ax.barh(dist_df["Segment"], dist_df["Distance"],
                          color=bar_colors, alpha=0.85, edgecolor="white", height=0.6)
            ax.set_xlabel("Distance to Centroid (lower = closer match)")
            ax.set_title(f"'{name}' is closest to {seg} centroid", fontweight="bold", fontsize=11)
            ax.axvline(dist_df["Distance"].min(), color="red", ls="--", lw=1.5, alpha=0.6)
            for bar, val in zip(bars, dist_df["Distance"]):
                ax.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
                       f"{val:.3f}", va="center", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ── LLM Message Chat Bubble ─────────────────────────────────────
        if use_llm:
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key

            with st.spinner("🤖 Generating personalised message with GPT-4o-mini..."):
                message = generate_message(seg, record)

            st.markdown("### 💬 Personalised Message")
            channel_icon = "📱" if channel=="WhatsApp" else "📧" if channel=="Email" else "💬"

            st.markdown(f"""
<div class="chat-bubble">
  <div class="chat-bubble-header">
    {channel_icon} {channel} message for <strong>{name}</strong>
    | Segment: <strong style="color:{color}">{seg}</strong>
    | Generated by GPT-4o-mini
  </div>
  {message}
</div>
""", unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            col_a.download_button("📥 Download Message", message,
                                  file_name=f"message_{customer_id}.txt",
                                  use_container_width=True)
            col_b.download_button("📊 Download Full Result",
                pd.DataFrame([{**record, "segment":seg, "confidence":conf,
                               "llm_message":message}]).to_csv(index=False),
                file_name=f"segment_{customer_id}.csv",
                mime="text/csv",
                use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — BATCH CSV UPLOAD
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Batch Segment All Customers from CSV")
    st.markdown("""
**How it works:**
1. Upload your customer CSV (required columns: `customer_id`, `recency_days`, `frequency`, `monetary_value`)
2. System assigns each customer to one of 7 segments using K-Means centroids
3. OpenAI generates a personalised message for each customer
4. Download results CSV → feed to WhatsApp Business API or Email platform
""")
    st.markdown("---")

    col1, col2 = st.columns([3,1])
    with col1:
        uploaded = st.file_uploader("Upload Customer CSV", type=["csv"])
    with col2:
        gen_msgs   = st.checkbox("Generate LLM Messages", True)
        openai_key2 = st.text_input("OpenAI Key", type="password",
                                     placeholder="sk-... (optional)")
        max_llm = st.number_input("Max LLM messages (0=all)", 0, 10000, 100,
                                   help="Set to 100 for demo to save API cost")

    if uploaded:
        df_in = pd.read_csv(uploaded)
        st.info(f"📊 Loaded **{len(df_in):,} customers** | {df_in.shape[1]} columns")

        with st.expander("Preview first 5 rows"):
            st.dataframe(df_in.head(5))

        if st.button("🚀 Segment All Customers + Generate Messages",
                     type="primary", use_container_width=True):

            if openai_key2:
                os.environ["OPENAI_API_KEY"] = openai_key2

            # Progress bar
            progress = st.progress(0, text="Assigning segments...")
            with st.spinner(f"Assigning segments to {len(df_in):,} customers..."):
                df_result = assign_batch(df_in)
            progress.progress(50, text="Segments assigned! Generating messages...")

            if gen_msgs:
                n_msg = min(max_llm, len(df_result)) if max_llm > 0 else len(df_result)
                df_msg  = df_result.head(n_msg).copy()
                messages = generate_batch_messages(df_msg)
                df_result["llm_message"] = ""
                df_result.loc[df_msg.index, "llm_message"] = messages
                if max_llm > 0 and max_llm < len(df_result):
                    df_result.loc[df_result.index[max_llm:], "llm_message"] = (
                        df_result.loc[df_result.index[max_llm:], "segment_name"].map(
                            lambda s: SEGMENT_STRATEGIES.get(s,{}).get("offer","Special offer for you")))

            df_result["processed_at"] = datetime.now().isoformat()
            progress.progress(100, text="Done! ✅")
            st.success(f"✅ {len(df_result):,} customers segmented!")

            # ── Summary Charts ──────────────────────────────────────────
            st.markdown("### 📊 Results Summary")
            seg_counts = df_result["segment_name"].value_counts()

            fig, axes = plt.subplots(1, 3, figsize=(22, 7))

            # Bar chart — customer count
            bar_colors = [COLORS.get(s,"#ccc") for s in seg_counts.index]
            bars = axes[0].bar(seg_counts.index, seg_counts.values,
                              color=bar_colors, alpha=0.85, edgecolor="white")
            axes[0].set_title("Customers per Segment", fontweight="bold", fontsize=12)
            axes[0].tick_params(axis="x", rotation=30)
            for bar, val in zip(bars, seg_counts.values):
                axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                            f"{val:,}\n({val/len(df_result)*100:.0f}%)",
                            ha="center", fontsize=8, fontweight="bold")

            # Pie — revenue share
            if "monetary_value" in df_result.columns:
                rev = df_result.groupby("segment_name")["monetary_value"].sum()
                axes[1].pie(rev.values, labels=rev.index, autopct="%1.1f%%",
                           colors=[COLORS.get(s,"#ccc") for s in rev.index],
                           startangle=90, wedgeprops={"edgecolor":"white","linewidth":2})
                axes[1].set_title("Revenue Share by Segment", fontweight="bold", fontsize=12)

            # Confidence distribution
            if "confidence" in df_result.columns:
                axes[2].hist(df_result["confidence"], bins=40,
                            color="#1F4E79", alpha=0.8, edgecolor="none")
                axes[2].set_xlabel("Assignment Confidence Score")
                axes[2].set_ylabel("Number of Customers")
                axes[2].set_title(f"Confidence Distribution\nMean: {df_result['confidence'].mean():.1%}",
                                 fontweight="bold", fontsize=12)
                axes[2].axvline(df_result["confidence"].mean(), color="red",
                               ls="--", lw=2, label=f"Mean")
                axes[2].legend()

            plt.suptitle(f"Segmentation Results — {len(df_result):,} Customers",
                        fontsize=14, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # ── Results Table ───────────────────────────────────────────
            st.markdown("### 📋 Sample Results (first 20)")
            display_cols = ["customer_id"]
            if "name"   in df_result.columns: display_cols.append("name")
            display_cols += ["segment_name","confidence","urgency","recommended_channel"]
            if "llm_message" in df_result.columns: display_cols.append("llm_message")

            st.dataframe(
                df_result[display_cols].head(20),
                use_container_width=True,
                column_config={
                    "segment_name":       st.column_config.TextColumn("Segment", width="medium"),
                    "confidence":         st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
                    "urgency":            st.column_config.TextColumn("Urgency", width="small"),
                    "recommended_channel":st.column_config.TextColumn("Channel", width="small"),
                    "llm_message":        st.column_config.TextColumn("AI Message", width="large"),
                }
            )

            # ── Segment breakdown table ─────────────────────────────────
            st.markdown("### 📊 Segment Breakdown")
            breakdown = df_result.groupby("segment_name").agg(
                customers=("customer_id","count"),
                avg_confidence=("confidence","mean"),
                urgency=("urgency","first"),
                channel=("recommended_channel","first"),
            ).round(3).sort_values("customers", ascending=False)

            if "monetary_value" in df_result.columns:
                rev_share = (df_result.groupby("segment_name")["monetary_value"].sum() /
                             df_result["monetary_value"].sum() * 100).round(1)
                breakdown["revenue_share_%"] = rev_share

            st.dataframe(breakdown, use_container_width=True)

            # ── Download buttons ────────────────────────────────────────
            st.markdown("### 📥 Download Results")
            col_d1, col_d2, col_d3 = st.columns(3)

            csv_full = df_result.to_csv(index=False)
            col_d1.download_button(
                "📥 Full Results CSV",
                csv_full,
                file_name=f"segmented_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", use_container_width=True
            )

            # At-Risk only
            at_risk_df = df_result[df_result["segment_name"].isin(["At Risk","Cannot Lose"])]
            col_d2.download_button(
                f"🚨 At-Risk + Cannot Lose ({len(at_risk_df):,})",
                at_risk_df.to_csv(index=False),
                file_name="urgent_customers.csv",
                mime="text/csv", use_container_width=True
            )

            # Champions only
            champ_df = df_result[df_result["segment_name"]=="Champions"]
            col_d3.download_button(
                f"⭐ Champions Only ({len(champ_df):,})",
                champ_df.to_csv(index=False),
                file_name="champions.csv",
                mime="text/csv", use_container_width=True
            )

            st.markdown("---")
            st.info("📤 **Next step:** Feed the downloaded CSV to your WhatsApp Business API or Email platform. Use the `llm_message` column as the message body and `recommended_channel` to decide the channel.")


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — SEGMENT OVERVIEW DASHBOARD
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Segment Overview — All 7 Segments")
    st.markdown("Strategy guide and profile for each customer segment")
    st.markdown("---")

    
# Load cluster profiles from S3 securely
try:
    REGION = os.getenv("AWS_REGION", "ap-southeast-2")
    BUCKET = os.getenv("S3_BUCKET", "customer-segmentation-2026-1")

    # Boto3 automatically retrieves credentials from AWS CLI
    # or the IAM role attached to the application.
    s3c = boto3.client(
        "s3",
        region_name=REGION
    )

    obj = s3c.get_object(
        Bucket=BUCKET,
        Key="models/cluster_profiles.csv"
    )

    profiles = pd.read_csv(
        io.BytesIO(obj["Body"].read())
    )

    # Metrics chart
    if "segment_name" in profiles.columns:

        plot_cols = [
            c for c in [
                "recency_days",
                "frequency",
                "monetary_value",
                "rfm_score"
            ]
            if c in profiles.columns
        ]

        if plot_cols:
            fig, axes = plt.subplots(
                1, len(plot_cols),
                figsize=(22, 6)
            )

            if len(plot_cols) == 1:
                axes = [axes]

            for ax, col in zip(axes, plot_cols):
                vals = profiles.set_index("segment_name")[col]

                colors = [
                    COLORS.get(s, "#ccc")
                    for s in vals.index
                ]

                ax.bar(
                    vals.index,
                    vals.values,
                    color=colors,
                    alpha=0.85,
                    edgecolor="white"
                )

                ax.set_title(
                    col.replace("_", " ").title(),
                    fontweight="bold"
                )

                ax.tick_params(
                    axis="x",
                    rotation=30
                )

            plt.suptitle(
                "Cluster Profiles from Trained K-Means++ Model",
                fontsize=14,
                fontweight="bold"
            )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

except Exception as e:
    st.warning(
        f"Unable to load cluster profiles from S3: {e}"
    )


