"""
llm_messages.py — LLM Message Generation for Each Segment
Author: Ashutosh Behera | Data Scientist | Bandhan Bank Ltd

Generates personalised marketing messages for each customer segment
using OpenAI GPT-4. Each message is unique per customer — not a template.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

SEGMENT_PROMPTS = {
    "Champions": """You are a premium customer experience manager at Big Basket.
Write a WhatsApp message for {name}.
They are our Champion customer — buying {frequency} times/year, spending Rs {spend:,.0f} annually.
Last purchased {recency} days ago. Favourite category: {category}.
Strategy: Give EXCLUSIVITY — no discount needed. They buy for quality not price.
Offer: Early access, VIP rewards, new product preview.
Tone: Premium, warm, appreciative. Make them feel special.
Length: Under 60 words. No discount mention.
Write only the message text — no labels or headers.""",

    "Loyal Customers": """You are a friendly relationship manager at Big Basket.
Write a WhatsApp message for {name}.
They are a Loyal customer — buying {frequency} times/year, spending Rs {spend:,.0f} annually.
Last purchased {recency} days ago. Favourite: {category}.
Strategy: Reward loyalty, encourage category expansion.
Offer: Loyalty points, bundle deal in a NEW category they have not tried.
Tone: Warm, rewarding, friendly.
Length: Under 65 words.
Write only the message text.""",

    "Potential Loyalists": """You are a helpful onboarding specialist at Big Basket.
Write a WhatsApp message for {name}.
They are a new customer — {frequency} orders so far, spending Rs {spend:,.0f}.
Last purchased {recency} days ago. Tried: {category}.
Strategy: Education and discovery — show what Big Basket can do for them.
Offer: 15% off first order in a new category.
Tone: Welcoming, helpful, excited to show them around.
Length: Under 65 words.
Write only the message text.""",

    "At Risk": """You are a caring retention specialist at Big Basket.
Write a WhatsApp message for {name}.
They USED to buy {frequency} times/year spending Rs {spend:,.0f}. Now silent for {recency} days.
Their usual items from {category} are waiting.
Strategy: Win-back with personalised urgency.
Offer: Rs 150 off — valid 48 hours only.
Tone: Warm, personal, slightly concerned — we miss you.
Mention their category by name.
Length: Under 70 words. Include expiry.
Write only the message text.""",

    "Cannot Lose": """You are a senior customer success manager at Big Basket.
Write a sincere WhatsApp message for {name}.
They were our BEST customer — {frequency} orders/year, Rs {spend:,.0f} annually.
Now silent for {recency} days. This is urgent — they are slipping away.
Strategy: Personal, premium win-back. Not a generic discount.
Offer: Personal call from their relationship manager + exclusive offer.
Tone: Sincere, caring, personal — not automated-feeling.
Length: Under 75 words.
Write only the message text.""",

    "Hibernating": """You are a re-engagement specialist at Big Basket.
Write an email subject line AND short body for {name}.
They bought {frequency} times before, spending Rs {spend:,.0f}, but last bought {recency} days ago.
Favourite: {category}.
Strategy: Gentle re-engagement — remind them what they are missing.
Offer: 20% off + free delivery on first order back.
Tone: Friendly, value-focused, no pressure.
Length: Subject under 10 words. Body under 60 words.
Write only the email content.""",

    "Lost Customers": """You are a sincere customer recovery specialist at Big Basket.
Write a final WhatsApp message for {name}.
They bought from us before ({frequency} orders, Rs {spend:,.0f}) but last bought {recency} days ago.
This is our last message before we stop contacting them.
Strategy: Genuine, humble. Acknowledge the gap. No pressure.
Offer: Rs 300 off — no conditions. Just come back.
Tone: Human, sincere, not like a marketing email.
Length: Under 80 words.
Write only the message text.""",
}

def generate_message(segment_name: str, customer: dict) -> str:
    """
    Generate a personalised marketing message for a customer using OpenAI.

    Input:  segment_name + customer dict with name, spend, recency etc.
    Output: personalised message string ready to send

    Cost: ~Rs 0.007 per customer (gpt-4o-mini)
    """
    if not OPENAI_API_KEY:
        return generate_fallback_message(segment_name, customer)

    prompt_template = SEGMENT_PROMPTS.get(segment_name, SEGMENT_PROMPTS["Hibernating"])
    prompt = prompt_template.format(
        name=customer.get("name", "Valued Customer"),
        frequency=int(customer.get("frequency", 12)),
        spend=float(customer.get("monetary_value", 15000)),
        recency=int(customer.get("recency_days", 30)),
        category=customer.get("favourite_category", "your favourites"),
    )

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.75,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI error: {e} — using fallback message")
        return generate_fallback_message(segment_name, customer)

def generate_fallback_message(segment_name: str, customer: dict) -> str:
    """
    Fallback message when OpenAI is not available.
    Template-based but still personalised with customer data.
    """
    name     = customer.get("name", "Valued Customer")
    category = customer.get("favourite_category", "your favourites")
    recency  = int(customer.get("recency_days", 30))

    fallbacks = {
        "Champions": f"Hi {name}! As one of our most valued customers, you get FIRST access to our new {category} collection — 48 hours before everyone else. Thank you for being the heart of Big Basket.",
        "Loyal Customers": f"Hi {name}! Your loyalty means everything to us. Here is a special bundle deal on {category} curated just for you. Shop now and earn double loyalty points this week!",
        "Potential Loyalists": f"Welcome, {name}! You loved {category} — did you know we deliver dairy and fresh produce within 2 hours? Here is 15% off your next category to explore.",
        "At Risk": f"Hi {name}, we have missed you! It has been {recency} days since your last {category} order. Your favourites are waiting — here is Rs 150 off, valid for 48 hours only.",
        "Cannot Lose": f"Hi {name}, we genuinely miss you as a customer. After {recency} days, we wanted to personally reach out. Here is our best offer — just for you. Can we chat?",
        "Hibernating": f"Hi {name}, Big Basket misses you! A lot has changed since your last visit {recency} days ago. Come back and enjoy 20% off + free delivery on your first order back.",
        "Lost Customers": f"Hi {name}, we know it has been a while ({recency} days). We would love to have you back. No conditions — here is Rs 300 off just to say hello again.",
    }
    return fallbacks.get(segment_name, f"Hi {name}, we have a special offer just for you!")

def generate_batch_messages(df_customers, segment_col="segment_name") -> list:
    """Generate messages for all customers in a DataFrame."""
    messages = []
    total = len(df_customers)
    for i, (_, row) in enumerate(df_customers.iterrows()):
        if (i+1) % 100 == 0:
            print(f"  Generated {i+1}/{total} messages...")
        msg = generate_message(row[segment_col], row.to_dict())
        messages.append(msg)
    return messages
