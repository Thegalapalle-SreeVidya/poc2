import streamlit as st
import joblib
import numpy as np
import time
import random
import warnings
import openai

# Suppress UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

# Define a custom client class to match the desired structure
class OpenAIClient:
    def __init__(self, api_key):
        openai.api_key = api_key

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        return openai.ChatCompletion.create(**kwargs)


# Initialize the client
client = OpenAIClient(api_key="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")


filename = 'support_vector_classifier_model.joblib'
loaded_model = joblib.load(filename)

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown(
    """
    <div style="text-align: center;">
        <h1>Credit Card Recommendation System 💳</h1>
        <p>Identify the best credit card tailored to your needs and preferences.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

st.markdown(
    """
    ### Step 1: Enter Customer Details
    Provide basic information about the customer to help us recommend the most suitable credit card.
    """
)

MAPPINGS = {
    "gender": {'Female': 0, 'Male': 1},
    "marital_status": {'Married': 0, 'Single': 1},
    "age_group": {'21-24': 0, '25-34': 1, '35-45': 2, '45+': 3},
    "city": {'Bengaluru': 0, 'Chennai': 1, 'Delhi NCR': 2, 'Hyderabad': 3, 'Mumbai': 4},
    "occupation": {'Business Owners': 0, 'Freelancers': 1, 'Government Employees': 2, 'Salaried IT Employees': 3, 'Salaried Other Employees': 4}
}

CLUSTER_DETAILS = {
    1: {
        "title": "Starter Cashback Credit Card Customers",
        "card_name": "Cashback Credit Card",
        "persona": "Young Budget-Conscious Professionals",
        "demographics": [
            "Most customers are aged 21-24, primarily male.",
            "Predominantly located in Hyderabad.",
            "Common occupation: Salaried IT Employees."
        ],
        "financial_profile": [
            "Average income is relatively low at ₹44,755.",
            "Monthly spending is minimal, averaging ₹2,123, with credit card spends at ₹1,438.",
            "Income Utilization: Low at 4.8%, indicating cautious spending habits.",
            "Credit Card Spends Percentage: High at 69.8%, indicating dependency on credit cards for purchases.",
            "Credit Score: Moderate, averaging 452."
        ],
        "offerings": [
            "Flat 5% cashback on online shopping, groceries, and utility payments.",
            "Waived annual fee for the first year.",
            "Rewards on every spend, redeemable for vouchers and discounts.",
            "No-cost EMI conversion for large purchases.",
            "Special fuel surcharge waiver at petrol pumps."
        ],
        "eligibility": [
            "Minimum income of ₹3,00,000 per year.",
            "Basic credit score of 450 or above.",
            "Focused on individuals with spending priorities on essentials and cashback benefits."
        ]
    },
    2: {
        "title": "Affluent High-Spenders",
        "card_name": "Premier Credit Card",
        "persona": "Affluent Professionals with High Spending Habits",
        "demographics": [
            "Customers are mostly aged 45+, primarily female.",
            "Predominantly located in Hyderabad.",
            "Common occupation: Salaried Other Employees."
        ],
        "financial_profile": [
            "High average income of ₹1,25,607.",
            "Significant monthly spending at ₹6,163, with very high credit card spends averaging ₹6,921.",
            "Income Utilization: Moderate at 4.9%.",
            "Credit Card Spends Percentage: Extremely high at 116%, indicating frequent credit card use for large purchases.",
            "Credit Score: High, averaging 633."
        ],
        "offerings": [
            "Unlimited complimentary airport lounge access worldwide.",
            "Reward points for every ₹100 spent, with accelerated rewards on travel and dining.",
            "Comprehensive travel insurance covering flight delays, lost baggage, and accidents.",
            "Global concierge services for reservations and bookings.",
            "Exclusive access to luxury brands and priority services."
        ],
        "eligibility": [
            "Minimum income of ₹15,00,000 per year.",
            "Credit score of 600 or above.",
            "Targeted at customers seeking exclusive lifestyle privileges and premium rewards."
        ]
    },
    3: {
        "title": "Wealthy Stable Spenders",
        "card_name": "Travel Credit Card",
        "persona": "Financially Stable Individuals with Balanced Habits",
        "demographics": [
            "Most customers are aged 35-45, primarily female.",
            "Predominantly located in Bengaluru.",
            "Common occupation: Salaried Other Employees."
        ],
        "financial_profile": [
            "High average income of ₹1,19,944.",
            "Monthly spending is substantial, averaging ₹7,870, with credit card spends at ₹3,652.",
            "Income Utilization: Slightly higher at 6.6%, indicating active financial engagement.",
            "Credit Card Spends Percentage: Moderate at 46.9%, showing balanced usage.",
            "Credit Score: High, averaging 631."
        ],
        "offerings": [
            "High reward points on travel and airline ticket bookings.",
            "Complimentary international lounge access.",
            "Zero foreign transaction fees for international purchases.",
            "Discounts on travel insurance and foreign currency conversion.",
            "Access to exclusive travel deals and partnerships with airlines and hotels."
        ],
        "eligibility": [
            "Minimum income of ₹10,00,000 per year.",
            "Credit score of 600 or above.",
            "Best suited for individuals who travel frequently and prioritize travel benefits."
        ]
    },
    4: {
        "title": "Frugal Single Professionals",
        "card_name": "Live Plus Credit Card",
        "persona": "Young Single Professionals Managing Modest Incomes",
        "demographics": [
            "Most customers are aged 21-24, primarily female.",
            "Predominantly located in Chennai.",
            "Common occupation: Salaried Other Employees."
        ],
        "financial_profile": [
            "Low average income of ₹42,698.",
            "Minimal monthly spending at ₹2,058, with credit card spends averaging ₹1,372.",
            "Income Utilization: Low at 4.9%.",
            "Credit Card Spends Percentage: High at 71.3%, reflecting reliance on credit cards.",
            "Credit Score: Moderate, averaging 444."
        ],
        "offerings": [
            "Welcome reward points for card activation.",
            "Discounted movie tickets and dining offers every month.",
            "Cashback on groceries and utility payments.",
            "Zero annual fee for the first year.",
            "Simple EMI options for affordable credit usage."
        ],
        "eligibility": [
            "Minimum income of ₹2,50,000 per year.",
            "Basic credit score of 450 or above.",
            "Ideal for individuals with a focus on savings and entry-level benefits."
        ]
    },
    5: {
        "title": "High-Income Balanced Professionals",
        "card_name": "Balanced Rewards Card",
        "persona": "Mature Professionals with Balanced Financial Management",
        "demographics": [
            "Most customers are aged 35-45, primarily female.",
            "Predominantly located in Delhi-NCR.",
            "Common occupation: Salaried Other Employees."
        ],
        "financial_profile": [
            "High average income of ₹1,15,540.",
            "Monthly spending is moderate at ₹4,217, with credit card spends at ₹2,579.",
            "Income Utilization: Lowest among clusters at 3.7%, showing efficient financial management.",
            "Credit Card Spends Percentage: Moderate at 63.1%, indicating healthy usage.",
            "Credit Score: Very high, averaging 672."
        ],
        "offerings": [
            "Flat 2% cashback on all purchases.",
            "Accelerated reward points on dining, travel, and online shopping.",
            "Fuel surcharge waivers across all fuel stations.",
            "Flexible redeemable points for travel, vouchers, or bill payments.",
            "Exclusive discounts on electronics and home appliances."
        ],
        "eligibility": [
            "Minimum income of ₹8,00,000 per year.",
            "Credit score of 650 or above.",
            "Designed for customers seeking balanced rewards across multiple spending categories."
        ]
    }
}


with st.form("my_form"):
    st.markdown(
        """
        #### Customer Demographics
        Fill in the details to help us analyze the customer profile.
        """
    )

    income = st.number_input(label='Monthly Income (INR)', step=100, min_value=25000, max_value=100000)
    spends = st.number_input(label='Monthly Spends (INR)', step=100, min_value=5000, max_value=50000)
    credit_card_spends = st.number_input(label='Monthly Credit Card Bill (INR)', step=100, min_value=0, max_value=20000, value=0, help="If you already have a credit card")
    credit_score = st.number_input(label='Credit Score', step=100, min_value=0, max_value=1000)
    income_utilization_perc = round((spends / income) * 100, 2)
    if credit_card_spends > 0:
        credit_card_spends_perc = round((credit_card_spends / spends) * 100, 2)
    else:
        credit_card_spends_perc = round(random.uniform(26, 50), 2)

    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

    marital_status = st.radio("Marital Status", ["Single", "Married"], horizontal=True)

    age_group = st.radio("Age Group you belong to", ["21-24", "25-34", "35-45", "45+"], horizontal=True)

    city = st.selectbox("City you live in", ["Bengaluru", "Chennai", "Delhi NCR", "Hyderabad", "Mumbai"])

    occupation = st.selectbox("Occupation", ["Business Owners", "Freelancers", "Government Employees", "Salaried IT Employees", "Salaried Other Employees"])

    aspirational_data = st.text_area("Tell us about your aspirations (e.g., planned purchases, goals, or lifestyle preferences)", "", help="Provide a brief description of your aspirations.")

    submitted = st.form_submit_button("Submit")

if submitted:

    with st.spinner("Processing..."):

        gender = MAPPINGS["gender"][gender]
        marital_status = MAPPINGS["marital_status"][marital_status]
        age_group = MAPPINGS["age_group"][age_group]
        city = MAPPINGS["city"][city]
        occupation = MAPPINGS["occupation"][occupation]
        DTI = round((spends * (credit_card_spends_perc / 100)) / income, 2)
        credit_limit_range = (income * DTI * 2.5, income * DTI * 5.5)
        data = [income, spends, credit_card_spends, income_utilization_perc, credit_card_spends_perc, credit_score, gender, marital_status, age_group, city, occupation]

        data = np.array(data).reshape(1, -1)
        pred = loaded_model.predict(data)

        # Simulate processing delay
        time.sleep(3)

    cluster_number = pred.item()+1
    cluster_info = CLUSTER_DETAILS.get(cluster_number, {})

    st.markdown("---")
    st.markdown(f"### Cluster {cluster_number}: {cluster_info['title']}")

    st.subheader(f"Recommended Credit Card: {cluster_info['card_name']}")
    st.markdown(f"**Persona:** {cluster_info['persona']}")

    st.markdown("#### Demographics:")
    for point in cluster_info["demographics"]:
        st.markdown(f"- {point}")

    st.markdown("#### Financial Profile:")
    for point in cluster_info["financial_profile"]:
        st.markdown(f"- {point}")

    st.markdown("#### Offerings:")
    for point in cluster_info["offerings"]:
        st.markdown(f"- {point}")

    st.markdown("#### Eligibility:")
    for point in cluster_info["eligibility"]:
        st.markdown(f"- {point}")

    st.markdown("#### Aspirational Input:")
    st.info(f"{aspirational_data}")
    
    llm_input = (
        f"Cluster {cluster_number} characteristics: {cluster_info['persona']} \n"
        f"Offerings: {', '.join(cluster_info['offerings'])} \n"
        f"Aspirational Input: {aspirational_data}"
    )

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a marketing assistant that creates personalized messages for credit card customer aqusition."},
            {"role": "user", "content": llm_input}
        ],
        max_tokens=150,
        temperature=0.7
    )

    # Get the response content
    llm_output = completion.choices[0].message.content.strip()

    # Display the output
    st.markdown("---")
    st.markdown(
        """
        ### Tailored Marketing Message
        Here is a customized message tailored to the customer:
        """
    )
    st.write(llm_output)


# Add content to the sidebar, including profile information
with st.sidebar:
    st.write("## New Credit Card Launch ")
    st.write("Explore our exclusive range of credit cards for different customer clusters.")
    st.write("*****")

st.markdown(
    """
    <div style="text-align: center;">
        <hr style="border: 1px solid #ddd; width: 100%;">
        <h5>This tool can be used by Customers or Banks to identify credit card recommendations.</h5>
    </div>
    """,
    unsafe_allow_html=True
)
