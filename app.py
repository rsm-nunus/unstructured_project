# %% [markdown]
# # Step 1: Import necessary libraries

# # %%
# !pip install sentence-transformers


# # %%
# ## Generate responess for these queries:

# !pip install transformers torch



# # %%
# !pip install --upgrade torch

# # %%
# ! pip install faiss-cpu


# # %%
# !pip install --upgrade transformers

# # %%
# pip install flask

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import warnings
warnings.filterwarnings("ignore")
import random
from transformers import pipeline
import faiss
import socket
from flask import Flask, request, jsonify, render_template
import json
from tqdm import tqdm
import re
import gc


# %% [markdown]
# # Step 2: Data Preparation

# %% [markdown]
# ## Read Retail Dataset

# %%
chunk_size = 10000  # Adjust as needed
chunks = []

for chunk in pd.read_csv('data/amazon_data.csv', chunksize=chunk_size):
    chunks.append(chunk)

final_df = pd.concat(chunks, ignore_index=True)
final_df.head()

# %% [markdown]
# ## Filtering the dataset for a subset of product categories - Cell Phones and Phone Accessories

# %%
final_df = final_df[final_df['category'].isin(['Cell Phones','Phone Accessories'])]

final_df.head()

# %%
final_df.shape

# %% [markdown]
# ## Filter for top 300 products

# %%
prod_details = pd.DataFrame(final_df['product_productId'].drop_duplicates().head(300))

# %%
final_df = pd.merge(final_df,prod_details,how='inner')

# %%
final_df.head()

# %% [markdown]
# ## Check for missing values

# %%
final_df.isnull().sum()

# %%
final_df.shape

# %% [markdown]
# ## Dropping irrelevant columns

# %%
final_df.columns

# %%
final_df.drop([ 'review_userId','review_profileName','review_summary','review_time'],inplace=True,axis=1)

# %%
final_df.head()

# %% [markdown]
# ## Create Products Dataframe

# %%
product_details = final_df[['product_title','category','product_productId']].drop_duplicates()

product_details.shape

# %%
product_details.head()

# %% [markdown]
# # Step 3: Feature Engineering

# %% [markdown]
# ## Generating Sentiment Scores for the Review column

# %%
def get_sentiment_textblob(text):
    sentiment_score = TextBlob(text).sentiment.polarity
    if sentiment_score > 0:
        return 'Positive'
    elif sentiment_score < 0:
        return 'Negative'
    else:
        return 'Neutral'

# %%
final_df['review_text'] = final_df['review_text'].astype(str)
final_df['text_sentiment'] = final_df['review_text'].apply(get_sentiment_textblob)

# %% [markdown]
# ## Create Embeddings for 'review_text' and 'product_title' columns

# %%
# Load a pre-trained model (BERT-based embeddings)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example: Encoding reviews and queries
final_df['review_embeddings'] = final_df['review_text'].apply(lambda x: model.encode(x))
# Generate embeddings for each product title
final_df['product_title_embedding'] = final_df['product_title'].apply(lambda x: model.encode(x).tolist())


# %% [markdown]
# ## Create Embeddings for product id
# 
# Approach for generating product id embeddings:
# - ASINs are unique 10-character identifiers that do not inherently encode product details.
# - Review Embeddings capture detailed information from customer reviews, including sentiment and features.
# - Averaging these embeddings provides a unified, noise-reduced representation of each product.
# - This aggregated representation can then be used in downstream tasks, such as product recommendations, clustering, or similarity searches.

# %%
# Function to average a list of embeddings
def average_embeddings(embeddings):
    # Convert the series of lists to a numpy array and calculate the mean along axis 0.
    arr = np.array(embeddings.tolist())
    return np.mean(arr, axis=0).tolist()

# Group by product ID and average the review embeddings
product_level_embeddings = final_df.groupby('product_productId')['review_embeddings'].agg(average_embeddings).reset_index()

print(product_level_embeddings)

final_df = final_df.merge(product_level_embeddings, on='product_productId', how='left', suffixes=('', '_avg'))


# %%
final_df.head()

# %% [markdown]
# ## One-hot Encoding

# %%
# One-hot encoding
# final_df = pd.get_dummies(final_df, columns=['category', 'text_sentiment'])

# %% [markdown]
# ## 'Helpfulness' column : Convert to Percentage format

# %%
# Function to convert "X/Y" format to percentage
def convert_helpfulness_ratio(value):
    try:
        num, denom = map(int, value.split("/"))
        return f"{(num / denom):.2f}" if denom != 0 else "0"
    except ValueError:
        return "Invalid"

# Apply conversion
final_df["helpfulness_percentage"] = final_df["review_helpfulness"].apply(convert_helpfulness_ratio)

final_df.head()


# %% [markdown]
# # Step 4: Query Data Generation

# %% [markdown]
# ### Generate Product Related Customer Queries

# # %%
# # Load text generation model (GPT-2)
# generator = pipeline("text-generation",model="gpt2")

# # %%
# # Query templates
# base_queries = [
#     "What are the features of PRODUCT_NAME?",
#     "Where can I buy PRODUCT_NAME at the best price?",
#     "Does PRODUCT_NAME have any issues?",
#     "Is PRODUCT_NAME worth buying?",
#     "How does PRODUCT_NAME compare to others in CATEGORY?",
#     "Are there any discounts available for PRODUCT_NAME?",
# ]


# # Function to generate diverse customer queries
# def generate_queries(product_name, category, base_query, num_variations=3):
#     prompt = f"Generate {num_variations} different ways a customer might ask: {base_query.replace('PRODUCT_NAME', product_name).replace('CATEGORY', category)}"

#     # Generate multiple variations
#     response = generator(prompt, max_length=500, do_sample=True, num_return_sequences=num_variations)

#     # Extract and clean responses
#     return [resp["generated_text"].split(":")[-1].strip() for resp in response]


# # %%
# # Generate queries for each product
# query_results = []

# for _, row in product_details.iterrows():
#     product_name = row["product_title"]
#     category = row["category"]

#     # Randomly select 3 base queries for variation
#     selected_queries = random.sample(base_queries, 3)

#     # Generate customer queries
#     generated_queries = []
#     for query in selected_queries:
#         generated_queries.extend(generate_queries(product_name, category, query))

#     # Store generated queries in a structured format
#     query_results.append([product_name, category, generated_queries])

# # Convert results to DataFrame
# query_final_df = pd.DataFrame(query_results, columns=["Product_Name", "Category", "Generated_Queries"]) 


# # %%
# pd.set_option('display.max_colwidth', None)
# query_final_df.head()

# # %%
# query_final_df.to_csv("query_final_data.csv", index=False)


# %% [markdown]
# ## Cleaning Generated Queries

# %%
## Going through the data and performing some manual cleaning and using ChatGPT help to load final query data


final_product_query_list = pd.read_csv('data/Final_Properly_Cleaned_Query_Data.csv')

final_product_query_list.head()

# %% [markdown]
# ## Generating Responses Data

# %%
# Load a free open-source language model that doesn't require authentication
# Use a simpler configuration without device_map or advanced settings
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with minimal settings to avoid the Accelerate dependency
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32  # Use regular float32 to avoid memory optimizations requiring accelerate
)

# Initialize the text generation pipeline with simpler parameters
chatbot = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)



# %%

# # Function to generate responses with print statements
# def generate_llm_response_with_print(query, product_name, category):
#     # Create a simple prompt - TinyLlama might not understand complex formatting
#     prompt = f"Customer question: '{query}' about {product_name} in {category} category. Provide a helpful response:"

#     # Generate response using LLM
#     try:
#         response = chatbot(
#             prompt, 
#             max_length=500, 
#             num_return_sequences=1
#         )[0]["generated_text"]
        
#         # Extract only the response part (everything after the prompt)
#         response = response.replace(prompt, "").strip()
        
#         # Truncate response if it's too long
#         if len(response) > 300:
#             response = response[:300] + "..."
            
#         # Print query and response for debugging
#         print("\n🔹 Query:", query)
#         print("🔹 Product:", product_name)
#         print("🔹 Category:", category)
#         print("\n💡 Generated Response:", response)
#         print("-" * 80)  # Separator for readability
        
#         return response
#     except Exception as e:
#         print(f"Error generating response: {e}")
#         return "We apologize, but we couldn't process your request at this moment. Please contact customer service for assistance."

# # Apply response generation to the first 5 rows only to test
# # You can remove the .head(5) to process the entire dataset
# sample_df = final_product_query_list
# sample_df["Response"] = sample_df.apply(
#     lambda row: generate_llm_response_with_print(row["Query"], row["Product_Name"], row["Category"]), axis=1
# )

# # Display the dataframe with queries and responses
# print("\n\n📊 Final Results:")
# print(sample_df[["Product_Name", "Query", "Response"]])

# %% [markdown]
# ## Create Product FAQ Dataset

# %%
# sample_df.to_csv("query_with_responses_data.csv", index=False)

# %% [markdown]
# ## Load Product FAQ Dataset

# %%
product_query_responses_data = pd.read_csv('data/query_with_responses_data.csv')

# %%
product_query_responses_data.head()

# %%
final_df.head()

# %% [markdown]
# # Step 5: Create Embeddings for Product FAQ Data

# %%
# Load a transformer model for encoding
model = SentenceTransformer("all-MiniLM-L6-v2")  # Small, fast & effective

# %%
# Encode queries
product_query_responses_data["query_embedding"] = product_query_responses_data["Query"].apply(lambda x: model.encode(x))

# Encode responses
product_query_responses_data["response_embedding"] = product_query_responses_data["Response"].apply(lambda x: model.encode(x))

# Convert to numpy arrays
query_embeddings = np.array(product_query_responses_data["query_embedding"].tolist()).astype("float32")
response_embeddings = np.array(product_query_responses_data["response_embedding"].tolist()).astype("float32")

# %%
product_query_responses_data.head()

# %%
# Create unique indices for each product
product_title_dict = {title: i for i, title in enumerate(product_query_responses_data["Product_Name"].unique())}

# Convert product IDs and titles to numerical indices
product_query_responses_data["product_title_index"] = product_query_responses_data["Product_Name"].map(product_title_dict)

# Generate random embeddings (Replace with trained embeddings in real case)
# embedding_dim = 8  # Choose based on model complexity
# product_query_responses_data["product_title_embedding"] = product_query_responses_data["product_title_index"].apply(lambda x: np.random.rand(embedding_dim).tolist())

product_query_responses_data.head()

# %% [markdown]
# ## One-hot encoding Product Query-Response Data

# %%
# One-hot encoding
# product_query_responses_data = pd.get_dummies(product_query_responses_data, columns=['Category'])

# %%
product_query_responses_data.head()

# %%
final_product_query_responses_data = product_query_responses_data

# %%
product_query_responses_data.columns

# %% [markdown]
# ## Convert Categorical dtypes to Numerical

# %%
# final_product_query_responses_data[['Category_Cell Phones','Category_Phone Accessories']] = final_product_query_responses_data[['Category_Cell Phones','Category_Phone Accessories']].astype(int)

# %%
# final_product_query_responses_data.head()

# %%
final_df.columns

# %%
prod_df = final_df

# %%
# prod_df[['category_Cell Phones',
#        'category_Phone Accessories', 'text_sentiment_Negative',
#        'text_sentiment_Neutral', 'text_sentiment_Positive']] = prod_df[['category_Cell Phones',
#        'category_Phone Accessories', 'text_sentiment_Negative',
#        'text_sentiment_Neutral', 'text_sentiment_Positive']].astype(int)


# %%
prod_df.head()

# %%
prod_df.columns

# %%
final_product_query_responses_data.head()

# %% [markdown]
# # Customer FAQ Data Generation

# %%
def generate_clean_responses(df, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                             batch_size=5, save_interval=10,
                             max_attempts=2, use_templates_only=False,
                             sample_n=None):  # New parameter for sampling rows
    """
    Generate clean, direct responses for Amazon FAQ entries using tone-adaptive templates,
    with optional model generation when available.
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing 'utterance', 'category', 'intent', and 'flags' columns.
        model_name (str): Name of the model to use.
        batch_size (int): Number of responses to process at once.
        save_interval (int): How often to save progress (in number of items).
        max_attempts (int): Maximum attempts to generate a valid response.
        use_templates_only (bool): If True, rely solely on templates without attempting model generation.
        sample_n (int, optional): If provided, process only a random sample of this many rows.
    
    Returns:
        pandas.DataFrame: (Sampled) DataFrame with added 'response' and 'tone_used' columns.
    """
    # Sample the DataFrame if requested
    if sample_n is not None:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
        print(f"Processing a sample of {sample_n} rows from the original DataFrame.")
    else:
        df = df.reset_index(drop=True)
    
    result_df = df.copy()
    result_df['response'] = None
    result_df['tone_used'] = None
    
    # Expanded template response dictionary with multiple variations for each tone
    template_responses = {
        "cancel_order": {
            "formal": [
                "To cancel your order, please visit Amazon.com and navigate to 'Your Orders'. Select the order you wish to cancel and click on the 'Cancel items' button. If the order has already entered the shipping process, you may need to request a return instead.",
                "You can cancel your order by accessing your Amazon account, selecting 'Your Orders', locating the specific order, and clicking 'Cancel items'. Please note that once an order has shipped, you'll need to proceed with the return process instead of cancellation."
            ],
            "casual": [
                "Need to cancel your order? No problem! Just hop onto Amazon.com, go to 'Your Orders', find the order you want to cancel, and hit the 'Cancel items' button. If it's already shipped, you'll need to return it instead, but that's super easy too!",
                "Canceling an order is quick and easy! Just head to 'Your Orders' on Amazon.com, find the order you want to cancel, and click the 'Cancel items' button. Already shipped? No worries - you can still return it when it arrives!"
            ],
            "concise": [
                "To cancel: Go to Amazon.com > 'Your Orders' > select order > 'Cancel items'. For shipped orders, request a return instead.",
                "Cancel order: Amazon.com > 'Your Orders' > find order > 'Cancel items'. If shipped, process a return."
            ],
            "polite": [
                "I'd be happy to assist you with canceling your order. Please visit Amazon.com and go to the 'Your Orders' section. From there, locate the order you wish to cancel and select the 'Cancel items' button. Should your order have already shipped, I'd recommend initiating the return process once you receive it.",
                "I understand you'd like to cancel an order. I'd be delighted to help with that. Please sign in to Amazon.com, navigate to 'Your Orders', find the specific order, and select the 'Cancel items' option. If your order has already shipped, please know that you can easily initiate a return through the same section."
            ]
        },
        "track_order": {
            "formal": [
                "To track your order, please sign in to your Amazon account and navigate to 'Your Orders'. Find the order you want to track and select 'Track Package'. You'll see the current status and estimated delivery date for your shipment.",
                "You can monitor your order status by logging into Amazon.com, selecting 'Your Orders' from the accounts menu, and clicking 'Track Package' next to the specific order. This will display detailed tracking information and the expected delivery date."
            ],
            "casual": [
                "Wondering where your package is? Easy! Just log into Amazon, click on 'Your Orders', find your purchase, and hit 'Track Package'. You'll see exactly where it is and when it's arriving!",
                "Want to check on your order? Super simple! Sign in to Amazon, go to 'Your Orders', find what you bought, and click 'Track Package'. You'll get all the details on where your goodies are and when they'll arrive!"
            ],
            "concise": [
                "Track order: Amazon.com > 'Your Orders' > find order > 'Track Package'.",
                "To track: Sign in > 'Your Orders' > select order > 'Track Package'."
            ],
            "polite": [
                "I'd be delighted to help you track your order. Please sign in to your Amazon account and navigate to the 'Your Orders' section. Once there, locate the order you're interested in and select the 'Track Package' option. This will provide you with the current location of your package and its anticipated delivery date.",
                "I understand you're looking to track your order. I'd be happy to assist with that. Please visit Amazon.com and sign in to your account. From there, select 'Your Orders', find the particular order you're inquiring about, and click on 'Track Package'. This will present you with detailed information about your package's journey and expected arrival."
            ]
        },
        "return_item": {
            "formal": [
                "To return an item, sign in to Amazon.com and go to 'Your Orders'. Locate the order containing the item you wish to return, select 'Return or replace items', and follow the guided process. You'll be able to select a return reason, preferred return method, and print a return label if needed.",
                "For returning an item, please access your Amazon account, navigate to 'Your Orders', find the specific purchase, and select 'Return or replace items'. The system will guide you through selecting a return reason and method. Most eligible items can be returned within 30 days of receipt."
            ],
            "casual": [
                "Need to send something back? No worries! Just go to Amazon.com, click on 'Your Orders', find what you want to return, and hit 'Return or replace items'. Follow the steps to tell us why you're returning it, choose how you want to send it back, and you're all set!",
                "Returns are super easy! Just log into your Amazon account, go to 'Your Orders', find the item, and click 'Return or replace items'. Follow the quick steps, and you'll have your return set up in no time. We'll even give you a shipping label to print!"
            ],
            "concise": [
                "To return: Amazon.com > 'Your Orders' > find item > 'Return or replace items' > follow prompts.",
                "Process return: Sign in > 'Your Orders' > select item > 'Return or replace items' > complete steps."
            ],
            "polite": [
                "I'd be happy to assist with your return. Please sign in to your Amazon account and visit the 'Your Orders' section. From there, please locate the item you wish to return, select 'Return or replace items', and follow the guided process. You'll be asked to provide a return reason and select your preferred return method. Please let me know if you need any further assistance with this process.",
                "I understand you'd like to return an item. I'd be delighted to help with that. Please visit Amazon.com and sign in to your account. Navigate to 'Your Orders', find the specific item you'd like to return, and select 'Return or replace items'. The system will guide you through selecting a reason for your return and your preferred return method. If you have any questions along the way, please don't hesitate to reach out for additional assistance."
            ]
        },
        "create_account": {
            "formal": [
                "To create an Amazon account, visit Amazon.com and click on 'Hello, Sign in' in the top right corner, then select 'New customer? Start here'. You'll need to provide your name, email address, and create a password. Once completed, you'll have access to all Amazon services including shopping, Prime, and more.",
                "You can register for an Amazon account by navigating to Amazon.com, selecting the 'Account & Lists' dropdown menu, and clicking on 'Start here' next to 'New to Amazon?'. The registration form will request your name, email address, and a secure password of your choosing."
            ],
            "casual": [
                "Creating an Amazon account is super easy! Just head to Amazon.com, click 'Account & Lists' at the top, and hit 'Start here' next to 'New to Amazon?'. Fill in your name, email, make a password, and you're all set to shop!",
                "Want to join Amazon? It's a breeze! Go to Amazon.com, click the 'Hello, Sign in' button at the top, and choose 'New customer? Start here'. Put in your details, create a password, and bam! You're ready to explore everything Amazon has to offer!"
            ],
            "concise": [
                "To register: Go to Amazon.com > 'Account & Lists' > 'New customer? Start here'. Enter details, create password, done.",
                "Create account: Amazon.com > 'Sign in' > 'New customer? Start here' > complete form."
            ],
            "polite": [
                "I'd be happy to help you create an Amazon account. Please visit Amazon.com and select 'Hello, Sign in' followed by 'New customer? Start here'. You'll then be guided through a simple registration process requiring your name, email address, and a secure password of your choosing.",
                "I understand you'd like to create an Amazon account. I'd be delighted to assist with that. Please navigate to Amazon.com and click on the 'Account & Lists' dropdown in the upper right corner. Then select 'Start here' next to where it says 'New to Amazon?'. You'll be asked to provide your name, email address, and to create a password. Once you've completed these steps, you'll have full access to Amazon's services."
            ]
        },
        "account_login": {
            "formal": [
                "To sign in to your Amazon account, visit Amazon.com and click on 'Hello, Sign in' in the top right corner. Enter the email address and password associated with your account. If you've forgotten your password, you can select the 'Forgot your password?' option to reset it.",
                "You can access your Amazon account by navigating to Amazon.com and selecting the 'Sign in' button located in the top navigation bar. Enter your registered email address and password in the designated fields. Should you need to recover your password, the 'Forgot your password?' link will guide you through the reset process."
            ],
            "casual": [
                "Logging in is easy! Go to Amazon.com, hit the 'Sign in' button in the top right, and enter your email and password. Forgot your password? No worries - just click the 'Forgot your password?' link to reset it!",
                "Need to sign in? Super simple! Head to Amazon.com, click the 'Hello, Sign in' button at the top, type in your email and password, and you're good to go! Can't remember your password? Just click 'Forgot your password?' and follow the steps to reset it!"
            ],
            "concise": [
                "To sign in: Amazon.com > 'Sign in' > enter email/password. For password reset, click 'Forgot your password?'",
                "Login: Amazon.com > 'Hello, Sign in' > enter credentials. Password help: Use 'Forgot your password?' link."
            ],
            "polite": [
                "I'd be delighted to assist with accessing your Amazon account. Please navigate to Amazon.com and select the 'Hello, Sign in' option located in the top right corner. You'll then need to enter your registered email address and password. Should you have trouble remembering your password, the 'Forgot your password?' link will guide you through the reset process.",
                "I understand you'd like to sign in to your Amazon account. I'd be happy to help with that. Please visit Amazon.com and click on the 'Sign in' button at the top of the page. From there, please enter the email address and password associated with your account. If you're having difficulty remembering your password, please use the 'Forgot your password?' option to reset it securely."
            ]
        }
    }
    
    # Generic templates for intents not specifically defined
    generic_templates = {
        "formal": [
            "To {intent_desc}, please visit Amazon.com and navigate to your account settings. From there, you can access the relevant section to manage your {intent_desc} preferences. For more specific assistance, you may contact Amazon Customer Service.",
            "You can {intent_desc} by accessing your Amazon account and selecting the appropriate option in your account settings. Amazon offers comprehensive self-service options for this request. If you require additional assistance, Amazon Customer Service representatives are available to help."
        ],
        "casual": [
            "Want to {intent_desc}? Easy! Just head to Amazon.com, go to your account settings, and you'll find options to handle that right there. Need more help? The customer service team is always ready to jump in!",
            "Looking to {intent_desc}? No problem! Log into Amazon.com, check out your account settings, and you'll find what you need. Still stuck? Our friendly customer service folks are just a click away!"
        ],
        "concise": [
            "To {intent_desc}: Visit Amazon.com > Account Settings > find relevant section. Or contact Customer Service.",
            "{intent_desc}: Amazon.com > Account > manage preferences. For help: Contact Support."
        ],
        "polite": [
            "I'd be happy to help you {intent_desc}. Please visit Amazon.com and access your account settings. There you'll find the appropriate options to manage this request. If you require any further assistance, please don't hesitate to contact our dedicated Customer Service team.",
            "I understand you'd like to {intent_desc}. I'd be delighted to assist with that. Please navigate to Amazon.com and sign in to your account. From there, please visit your account settings where you'll find the relevant options. Should you need additional help, our Customer Service representatives are always available to provide personalized assistance."
        ]
    }
    
    # Define tone descriptions for output
    tone_descriptions = {
        "formal": "Professional and thorough",
        "casual": "Friendly and conversational",
        "concise": "Brief and direct",
        "polite": "Courteous and respectful"
    }
    
    # Load model only if not using template-only mode
    model = None
    tokenizer = None
    if not use_templates_only:
        print("Loading model and tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            print(f"Successfully loaded {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print("Using template responses only.")
            use_templates_only = True
    else:
        print("Template-only mode selected. Skipping model loading.")
    
    if model is not None:
        print(f"Model loaded on {'GPU' if torch.cuda.is_available() else 'CPU'}")
    else:
        print("Running in template-only mode")
    
    def determine_tone_style(flags):
        flags_str = ''.join(flags) if isinstance(flags, list) else flags
        if 'P' in flags_str:
            return "polite"
        if 'Q' in flags_str:
            return "casual"
        if 'C' in flags_str and 'B' not in flags_str:
            return "formal"
        if 'B' in flags_str and 'C' not in flags_str:
            return "concise"
        return "formal"
    
    def interpret_intent(intent, utterance):
        """Extract a human-readable intent description using mappings."""
        intent_desc = intent.replace('_', ' ')
        intent_mappings = {
            "cancel_order": "cancel your order",
            "track_order": "track your order",
            "return_item": "return an item",
            "create_account": "create an Amazon account",
            "account_login": "sign in to your account",
            "edit_account": "edit your account information",
            "payment_issue": "report a payment issue",
            "newsletter_subscription": "manage your newsletter subscription",
            "check_refund_policy": "check your refund policy",
            "track_refund": "track your refund status",
            "review": "submit a review",
            "place_order": "place an order",
            "get_invoice": "get your invoice",
            "complaint": "file a complaint",
            "contact_customer_service": "contact customer service",
            "change_order": "change your order",
            "recover_password": "recover your password",
            "delivery_options": "check delivery options",
            "switch_account": "switch your account",
            "registration_problems": "resolve registration problems",
            "check_payment_methods": "check available payment methods",
            "set_up_shipping_address": "set up a shipping address",
            "delete_account": "delete your account",
            "contact_human_agent": "contact a human agent",
            "check_cancellation_fee": "check cancellation fees",
            "delivery_period": "check delivery period"
        }
        return intent_mappings.get(intent, intent_desc)
    
    def generate_response(intent, tone, utterance):
        if intent in template_responses:
            templates = template_responses[intent].get(tone, template_responses[intent]["formal"])
            response_text = random.choice(templates)
        else:
            intent_desc = interpret_intent(intent, utterance)
            templates = generic_templates.get(tone, generic_templates["formal"])
            template_choice = random.choice(templates)
            response_text = template_choice.format(intent_desc=intent_desc)
        print(f"Generated template response: {response_text}")
        return response_text

    def clean_output(text):
        text = re.sub(r'^(AI:|Assistant:|Amazon:|Bot:|Q:).*?\n', '', text, flags=re.IGNORECASE)
        # matches = re.findall(r'(?:^|\n)([A-Z].*?(?:\.|!|\?))(?:\n|$)', text)
        # if matches:
        #     return matches[0].strip()
        # if len(text) > 200:
        #     truncation_point = text.rfind('.', 0, 200)
        #     if truncation_point > 0:
        #         return text[:truncation_point + 1].strip()
        return text.strip()

    def model_generate_response(utterance, intent, tone):
        if model is None or tokenizer is None:
            return None
        prompt = f"""You are an Amazon customer service assistant.
Please provide a {tone} and relevant response to the following customer question. Respond in one concise paragraph and do not include extra commentary or repeated instructions.
Do NOT instruct the customer to use our chat service, since this is already a chat interface.
Customer: {utterance}
Intent: {intent}
Response:"""
        print(f"Model prompt:\n{prompt}")
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.6,
                    top_p=0.85,
                    do_sample=True,
                    repetition_penalty=1.2
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Response:" in response:
                response = response.split("Response:")[-1].strip()
            response = clean_output(response)
            print(f"Model generated response: {response}")
            return response
        except Exception as e:
            print(f"Error during model generation: {e}")
            return None

    print("Starting processing of DataFrame rows...")
    for i, row in result_df.iterrows():
        print(f"\nProcessing row {i+1}/{len(result_df)}")
        flags = row['flags']
        tone = determine_tone_style(flags)
        intent = row['intent']
        utterance = row['utterance']
        print(f"Flags: {flags} | Determined tone: {tone} | Intent: {intent}")
        print(f"Utterance: {utterance}")
        
        response = None
        if not use_templates_only:
            for attempt in range(max_attempts):
                print(f"Model generation attempt {attempt+1}...")
                response = model_generate_response(utterance, intent, tone)
                if response and len(response) > 20:
                    print("Model generated a valid response.")
                    break
                else:
                    print("Model response not valid, trying again...")
        if not response:
            print("Falling back to template response.")
            response = generate_response(intent, tone, utterance)
        
        print(f"Final response for row {i+1}: {response}")
        result_df.at[i, 'response'] = response
        result_df.at[i, 'tone_used'] = tone

        if (i + 1) % save_interval == 0 or i == len(result_df) - 1:
            result_df.iloc[:i+1].to_csv('amazon_faq_responses_progress.csv', index=False)
            print(f"Checkpoint saved at row {i+1}/{len(result_df)}")
        
        if model is not None and torch.cuda.is_available() and (i + 1) % batch_size == 0:
            torch.cuda.empty_cache()
            gc.collect()
            print("Cleared GPU cache.")

    result_df.to_csv('amazon_faq_with_tone_adaptive_responses.csv', index=False)
    print("Final results saved to 'amazon_faq_with_tone_adaptive_responses.csv'.")
    if model is not None:
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        print("Cleaned up model and tokenizer from memory.")

    return result_df

# %%
# df = pd.read_csv("data/20000-Utterances-Training-dataset-for-chatbots-virtual-assistant-Bitext-sample/20000-Utterances-Training-dataset-for-chatbots-virtual-assistant-Bitext-sample/20000-Utterances-Training-dataset-for-chatbots-virtual-assistant-Bitext-sample.csv")
# results = generate_clean_responses(df, sample_n=100)

# %% [markdown]
# # Generate Embeddings for Customer FAQ Data

# %% [markdown]
# ## Load Customer FAQ data

# %%
# Load your FAQ data from CSV (adjust the column names as needed)
faq_df = pd.read_csv('data/amazon_faq_with_tone_adaptive_responses.csv')  
print("FAQ data loaded. Sample:")
print(faq_df.head())

# Initialize SentenceTransformer with a model of your choice (e.g., 'all-MiniLM-L6-v2')
model = SentenceTransformer('all-MiniLM-L6-v2')

# %% [markdown]
# ## Compute Embeddings for Queries and responses

# %%

# Precompute embeddings for each FAQ question
faq_questions = faq_df['utterance'].tolist()
faq_embeddings = model.encode(faq_questions, convert_to_numpy=True)
print("Embeddings generated for FAQ questions.")


# %%
# Precompute embeddings for each FAQ response
faq_responses = faq_df['response'].tolist()
faq_resp_embeddings = model.encode(faq_responses, convert_to_numpy=True)
print("Embeddings generated for FAQ responses.")


# %%
faq_df['query_embeddings'] = list(faq_embeddings)
faq_df['response_embeddings']= list(faq_resp_embeddings)
faq_df.head()

# %% [markdown]
# # Step 6: Implementing FAISS (Facebook AI Similarity Search)

# %% [markdown]
# ## FAISS for Customer FAQ's

# %%
# Normalize the embeddings if you plan to use cosine similarity
faq_embeddings = faq_embeddings / np.linalg.norm(faq_embeddings, axis=1, keepdims=True)
faq_resp_embeddings = faq_resp_embeddings / np.linalg.norm(faq_resp_embeddings, axis=1, keepdims=True)
faq_embeddings

# %%
dimension = faq_embeddings.shape[1]  # Embedding dimension
index = faiss.IndexFlatIP(dimension)  # Using inner product (with normalized vectors, this is cosine similarity)
index.add(faq_embeddings)  # Add embeddings to the index

print("FAISS index built with {} embeddings.".format(index.ntotal))

# %% [markdown]
# ## FAISS for Product FAQ's

# %%
# Convert embeddings to numpy arrays
query_embeddings = np.array(final_product_query_responses_data["query_embedding"].tolist()).astype("float32")
response_embeddings = np.array(final_product_query_responses_data["response_embedding"].tolist()).astype("float32")
product_embeddings = np.array(prod_df["product_title_embedding"].tolist()).astype("float32")
product_id_embeddings = np.array(prod_df["review_embeddings_avg"].tolist()).astype("float32")

# Store product IDs and titles separately for retrieval
product_ids = prod_df["product_productId"].tolist()
product_titles = prod_df["product_title"].tolist()

# Get dimensions from each embedding type
query_dim = query_embeddings.shape[1]    # 384
product_dim = product_embeddings.shape[1]  # 8

print(f"Query embeddings shape: {query_embeddings.shape}")
print(f"Response embeddings shape: {response_embeddings.shape}")
print(f"Product embeddings shape: {product_embeddings.shape}")
print(f"Product ID embeddings shape: {product_id_embeddings.shape}")

# Create FAISS indexes with the correct dimensions for each type
query_index = faiss.IndexFlatL2(query_dim)
response_index = faiss.IndexFlatL2(query_dim)  # Same as query dimension
product_index = faiss.IndexFlatL2(product_dim)
product_id_index = faiss.IndexFlatL2(product_dim)

# Add embeddings to FAISS
query_index.add(query_embeddings)
response_index.add(response_embeddings)
product_index.add(product_embeddings)
product_id_index.add(product_id_embeddings)

# Save FAISS indexes
faiss.write_index(query_index, "query_index.faiss")
faiss.write_index(response_index, "response_index.faiss")
faiss.write_index(product_index, "product_index.faiss")
faiss.write_index(product_id_index, "product_id_index.faiss")

# Save product ID & title mapping separately for retrieval
product_metadata = pd.DataFrame({"product_productId": product_ids, "product_title": product_titles})

# %% [markdown]
# # Response Retrieval Functions

# %% [markdown]
# ## Product FAQ's

# %%
### SENTIMENT FACTORS:


# Function to analyze sentiment of review text
def analyze_sentiment(review_text):
    if not review_text or not isinstance(review_text, str):
        return {"label": "Neutral", "score": 0.5}  # Default to Neutral
        
    try:
        result = sentiment_analyzer(review_text)
        return result[0]  # Returns dict with 'label' and 'score'
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return {"label": "Neutral", "score": 0.5}  # Default fallback

# Improved sentiment analysis that considers both text and score
def compute_sentiment(review_text, review_score):
    """Compute sentiment using both the review text and numerical score."""
    # Use the sentiment analyzer for text analysis
    text_sentiment = analyze_sentiment(review_text)
    
    # Determine score-based sentiment
    if review_score >= 4.0:
        score_sentiment = {"label": "Positive", "score": review_score / 5}
    elif review_score <= 2.0:
        score_sentiment = {"label": "Negative", "score": review_score / 5}
    else:
        score_sentiment = {"label": "Neutral", "score": review_score / 5}
    
    # Weighted combination (give more weight to the review score)
    final_score = 0.3 * text_sentiment["score"] + 0.7 * score_sentiment["score"]
    
    if final_score >= 0.65:
        label = "Positive"
    elif final_score <= 0.4:
        label = "Negative"
    else:
        label = "Neutral"
        
    return {"label": label, "score": final_score}


# %%
def project_embeddings(source_embedding, target_dim):
    """Project embeddings from source dimension to target dimension using PCA-like approach."""
    from sklearn.decomposition import PCA
    import numpy as np
    
    # Ensure the source embedding is a 2D numpy array
    if isinstance(source_embedding, list):
        source_embedding = np.array(source_embedding).astype('float32')
    
    # Reshape if it's a 1D array
    if source_embedding.ndim == 1:
        source_embedding = source_embedding.reshape(1, -1)
    
    # If dimensions already match, return as is
    if source_embedding.shape[1] == target_dim:
        return source_embedding
    
    # Handle dimension mismatch
    if source_embedding.shape[1] > target_dim:
        # Use PCA to reduce dimensions
        pca = PCA(n_components=target_dim)
        try:
            reduced = pca.fit_transform(source_embedding).astype('float32')
            return reduced
        except Exception as e:
            print(f"PCA dimensionality reduction failed: {e}")
            # Fallback: simple truncation
            return source_embedding[:, :target_dim].astype('float32')
    else:
        # Pad with zeros to reach target dimension
        padded = np.zeros((source_embedding.shape[0], target_dim), dtype='float32')
        padded[:, :source_embedding.shape[1]] = source_embedding
        return padded

# %%
def retrieve_best_product(user_query, product_metadata, prod_df):
    """Retrieve product using NLP similarity techniques for better semantic matching."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import re
    
    # Parse query for important information
    query_normalized = user_query.lower()
    
    # Extract price constraints if present
    price_limit = None
    price_patterns = [
        r'under\s+\$?(\d+\.?\d*)',
        r'below\s+\$?(\d+\.?\d*)',
        r'less than\s+\$?(\d+\.?\d*)',
        r'cheaper than\s+\$?(\d+\.?\d*)',
        r'under\s+(\d+\.?\d*)\s+dollars',
        r'below\s+(\d+\.?\d*)\s+dollars'
    ]
    
    for pattern in price_patterns:
        match = re.search(pattern, query_normalized)
        if match:
            price_limit = float(match.group(1))
            print(f"📊 Detected price limit: ${price_limit}")
            break
    
    # Extract product category information
    category_keywords = {
        "phone": ["phone", "cell", "mobile", "smartphone", "calling"],
        "camera": ["camera", "photo", "picture", "photography"],
        "headset": ["headset", "earphone", "headphone", "earpiece", "earbud"],
        "accessory": ["accessory", "case", "cover", "protector", "holder", "clip"]
    }
    
    detected_categories = []
    for category, keywords in category_keywords.items():
        if any(keyword in query_normalized for keyword in keywords):
            detected_categories.append(category)
    
    if detected_categories:
        print(f"📂 Detected categories: {', '.join(detected_categories)}")
    
    # Extract features/attributes of interest
    feature_keywords = {
        "color": ["pink", "black", "white", "silver", "blue", "red", "gold"],
        "brand": ["samsung", "nokia", "motorola", "jabra", "apple", "google"],
        "feature": ["mp3", "music", "camera", "video", "unlocked", "bluetooth", "wireless"]
    }
    
    detected_features = {}
    for feature_type, keywords in feature_keywords.items():
        found = [keyword for keyword in keywords if keyword in query_normalized]
        if found:
            detected_features[feature_type] = found
    
    if detected_features:
        print(f"🔍 Detected features: {detected_features}")
    
    # Step 1: Filter products based on extracted constraints
    filtered_df = prod_df.copy()
    
    # Apply category filter if detected
    if detected_categories and "category" in filtered_df.columns:
        category_mask = filtered_df["category"].str.lower().apply(
            lambda x: any(category in x.lower() for category in detected_categories) if isinstance(x, str) else False
        )
        filtered_df = filtered_df[category_mask]
    
    # Apply brand filter if detected
    if "brand" in detected_features and "product_title" in filtered_df.columns:
        brand_mask = filtered_df["product_title"].str.lower().apply(
            lambda x: any(brand in x.lower() for brand in detected_features["brand"]) if isinstance(x, str) else False
        )
        filtered_df = filtered_df[brand_mask]
    
    # Apply price filter if detected (assuming product_price column exists)
    if price_limit is not None and "product_price" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["product_price"] <= price_limit]
    
    # If no products match our filters, revert to original dataframe
    if filtered_df.empty:
        print("⚠️ No products match filters, reverting to full product list")
        filtered_df = prod_df.copy()
    
    # Step 2: Calculate semantic similarity between query and products
    # Prepare corpus for TF-IDF calculation
    product_titles = filtered_df["product_title"].fillna("").tolist()
    
    # If no products in filtered list, use a fallback
    if not product_titles:
        print("⚠️ No product titles available after filtering")
        return None, None, None, {"label": "Neutral", "score": 0.5}, "Unknown"
    
    # Create corpus with query and product titles
    corpus = [query_normalized] + product_titles
    
    try:
        # Calculate TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Calculate similarity between query and all products
        query_vector = tfidf_matrix[0:1]
        product_vectors = tfidf_matrix[1:]
        
        # Get similarity scores
        similarity_scores = cosine_similarity(query_vector, product_vectors)[0]
        
        # Get index of most similar product
        best_product_idx = np.argmax(similarity_scores)
        best_similarity = similarity_scores[best_product_idx]
        
        print(f"🔢 Best match similarity score: {best_similarity:.4f}")
        
        # Get product details
        matched_product_row = filtered_df.iloc[best_product_idx]
        matched_product_id = matched_product_row["product_productId"]
        matched_product_title = matched_product_row["product_title"]
        
        # Get review data
        matched_reviews = filtered_df.loc[filtered_df["product_productId"] == matched_product_id, "review_text"].dropna().tolist()
        matched_scores = filtered_df.loc[filtered_df["product_productId"] == matched_product_id, "review_score"].dropna().tolist()
        
        if matched_reviews and matched_scores:
            review_text = matched_reviews[0]
            review_score = float(matched_scores[0])
            
            # Sentiment analysis based on review score
            if review_score >= 4.0:
                sentiment_info = {"label": "Positive", "score": 0.85}
            elif review_score <= 2.0:
                sentiment_info = {"label": "Negative", "score": 0.25}
            else:
                sentiment_info = {"label": "Neutral", "score": 0.5}
            
            category = matched_product_row["category"] if "category" in matched_product_row else "Cell Phones"
            
            print(f"✅ Found best matching product: {matched_product_title}")
            return matched_product_id, matched_product_title, review_text, sentiment_info, category
        
    except Exception as e:
        print(f"⚠️ Error during similarity calculation: {e}")
    
    # Fallback to simple keyword matching if TF-IDF fails
    print("⚠️ Falling back to keyword matching")
    
    # Try to find Samsung products as fallback
    fallback_products = product_metadata[
        product_metadata["product_title"].str.lower().str.contains("samsung", na=False)
    ]
    
    if not fallback_products.empty:
        matched_product_id = fallback_products["product_productId"].iloc[0]
        matched_product_title = fallback_products["product_title"].iloc[0]
        
        # Get product details
        product_row = prod_df[prod_df["product_productId"] == matched_product_id]
        
        if not product_row.empty:
            matched_reviews = product_row["review_text"].dropna().tolist()
            matched_scores = product_row["review_score"].dropna().tolist()
            
            if matched_reviews and matched_scores:
                review_text = matched_reviews[0]
                review_score = float(matched_scores[0])
                
                # Simple sentiment analysis
                if review_score >= 4.0:
                    sentiment_info = {"label": "Positive", "score": 0.85}
                elif review_score <= 2.0:
                    sentiment_info = {"label": "Negative", "score": 0.25}
                else:
                    sentiment_info = {"label": "Neutral", "score": 0.5}
                
                category = product_row["category"].iloc[0] if "category" in product_row.columns else "Cell Phones"
                
                print(f"⚠️ Using fallback product: {matched_product_title}")
                return matched_product_id, matched_product_title, review_text, sentiment_info, category
    
    print("⚠️ No matching product found")
    return None, None, None, {"label": "Neutral", "score": 0.5}, "Unknown"

# %%
def retrieve_best_response(user_query, final_product_query_responses_data, prod_df, product_metadata):
    """Retrieve best-matching stored query and response without using embeddings."""
    
    # Normalize user query
    user_query_normalized = user_query.lower().strip()
    
    # Default values
    best_query = "No relevant match found"
    best_response = "I'll find products based on your request instead."
    is_matched = False
    
    try:
        # Find exact matches for the query
        matched_rows = final_product_query_responses_data[
            final_product_query_responses_data["Query"].str.lower().str.strip() == user_query_normalized
        ]
        
        if not matched_rows.empty:
            print(f"✅ Found exact match in dataset!")
            matched_row = matched_rows.iloc[0]
            best_query = str(matched_row["Query"])
            best_response = str(matched_row.get("Response", "No stored response available"))
            is_matched = True
    except Exception as e:
        print(f"Error finding exact match: {e}")
    
    # Get product directly by name matching, not using embeddings
    matched_product_id, matched_product_title, review_text, sentiment_info, category = retrieve_best_product(
        user_query, product_metadata, prod_df
    )
    
    return best_query, best_response, review_text, sentiment_info, is_matched, matched_product_id, matched_product_title, category

# %%
# Improved alternative suggestions that avoid duplicates and ensure quality
def generate_alternative_suggestions(user_query, matched_product_id, prod_df, exclude_ids=None):
    """Generate unique, high-quality alternative product suggestions."""
    
    # ✅ Ensure exclude_ids is initialized correctly
    if exclude_ids is None:
        exclude_ids = set()
    exclude_ids.add(matched_product_id)  # Use a set to prevent duplicate entries
    
    # ✅ Fetch product details to get category
    product_row = prod_df[prod_df["product_productId"] == matched_product_id]
    
    if product_row.empty or "category" not in product_row.columns:
        return "No suitable alternatives found.", []
    
    matched_category = product_row["category"].iloc[0]
    
    # ✅ Ensure review_score is numeric and handle missing values
    prod_df["review_score"] = pd.to_numeric(prod_df["review_score"], errors='coerce')  # Convert to numeric
    prod_df = prod_df.dropna(subset=["review_score"])  # Drop rows where review_score is NaN
    
    # ✅ Get top-rated alternatives from the same category, excluding already suggested products
    alternatives = prod_df[
        (prod_df["category"] == matched_category) & 
        (prod_df["review_score"] >= 4.0) & 
        (~prod_df["product_productId"].isin(exclude_ids))
    ][["product_productId", "product_title", "review_score"]].drop_duplicates(subset=["product_title"]).sort_values(
        by="review_score", ascending=False
    ).head(3)
    
    if alternatives.empty:
        return "No suitable alternatives found.", []
    
    # ✅ Format alternatives for chatbot response
    alternative_recommendations = "\n".join([
        f"- {row['product_title']} (⭐ {row['review_score']}/5)" 
        for _, row in alternatives.iterrows()
    ])
    
    # ✅ Return both text and a unique list of alternative product IDs
    return alternative_recommendations, list(alternatives["product_productId"].unique())


# %%
def generate_chatbot_response(user_query, matched_query, response, product_name, review_text, sentiment_info, is_matched, matched_product_id, category, prod_df):
    """Generate a relevant and coherent response to the user query."""
    
    # IMPORTANT: If we have an exact match, use that response directly
    if is_matched and response and response != "No stored response available":
        print("✅ Using exact match response from database")
        return response
    
    # If no exact match or no valid stored response, generate a new one
    print("⚠️ No exact match response found, generating new response")
    
    # Ensure product_name is valid before generating a response
    if not product_name or product_name == "Unknown":
        return f"I couldn't find any products matching your query about '{user_query}'. Could you provide more details about what you're looking for?"
    
    # Ensure sentiment label exists before using it
    sentiment_label = sentiment_info.get("label", "Neutral")
    
    # Generate appropriate sentiment message based on sentiment label
    if sentiment_label == "Negative":
        sentiment_message = f"⚠️ The reviews for {product_name} suggest that it has issues."
        # Get alternatives, excluding the negative product
        alternative_recommendations, _ = generate_alternative_suggestions(user_query, matched_product_id, prod_df)
        suggestion = f"Here are some better alternatives in the {category} category:\n{alternative_recommendations}" if alternative_recommendations else "Unfortunately, I couldn't find better alternatives."
    elif sentiment_label == "Positive":
        sentiment_message = f"✅ {product_name} has received good reviews. It might be a great option for you!"
        # Still provide alternatives for comparison
        alternative_recommendations, _ = generate_alternative_suggestions(user_query, matched_product_id, prod_df)
        suggestion = f"Here are some other highly-rated options in the {category} category if you want to compare:\n{alternative_recommendations}" if alternative_recommendations else "No additional alternatives available at this time."
    else:
        sentiment_message = f"{product_name} has mixed reviews."
        alternative_recommendations, _ = generate_alternative_suggestions(user_query, matched_product_id, prod_df)
        suggestion = f"Here are some alternatives in the {category} category that might interest you:\n{alternative_recommendations}" if alternative_recommendations else "Unfortunately, no relevant alternatives were found."
    
    # Truncate review text if too long, ensuring it's not None
    review_preview = review_text[:100] + "..." if review_text and isinstance(review_text, str) and len(review_text) > 100 else (review_text if review_text else "No customer reviews available.")
    
    # Address specific query types
    if "discount" in user_query.lower() or "deal" in user_query.lower() or "sale" in user_query.lower():
        discount_info = f"Regarding discounts for {product_name}, I don't have current promotion information. I recommend checking with retailers directly for the latest deals and offers."
        context_specific = discount_info
    elif "price" in user_query.lower() or "cost" in user_query.lower() or "how much" in user_query.lower():
        price_info = f"For the most up-to-date pricing on {product_name}, I recommend checking with authorized retailers. Prices may vary based on promotions and availability."
        context_specific = price_info
    elif "compare" in user_query.lower():
        context_specific = f"When comparing {product_name} with alternatives, customers particularly mention its features. {suggestion}"
    elif "feature" in user_query.lower() or "specification" in user_query.lower() or "spec" in user_query.lower():
        context_specific = f"{product_name} comes with various features that customers have reviewed. For detailed specifications, I recommend checking the manufacturer's website."
    else:
        context_specific = ""
    
    # Create a clean, user-friendly response based on query context
    response = f"For your search about {category} products, I recommend checking out {product_name}.\n\n{sentiment_message}\n\nHere's what a customer said: \"{review_preview}\""
    
    # Add context-specific information if available
    if context_specific:
        response += f"\n\n{context_specific}"
    
    # Add suggestions if not already included in context_specific
    if "alternatives" not in response.lower() and suggestion and "Here are some" not in context_specific:
        response += f"\n\n{suggestion}"
    
    return response

# %%
def get_product_recommendations(user_input):
    """Complete recommendation flow without using embeddings."""
    
    global final_product_query_responses_data
    
    # Load data and indexes
    prod_df = final_df
    
    # Load product metadata
    product_metadata = pd.DataFrame({
        "product_productId": prod_df["product_productId"].tolist(),
        "product_title": prod_df["product_title"].tolist()
    })
    
    # Ensure the "Query" column exists before processing
    if "Query" not in final_product_query_responses_data.columns:
        raise KeyError("⚠️ Column 'Query' not found in final_product_query_responses_data. Check column names.")
    
    # Drop NaN values and convert to string type
    final_product_query_responses_data["Query"] = final_product_query_responses_data["Query"].fillna("").astype(str)
    
    # Get query match and response without using embeddings
    matched_query, matched_response, review_text, sentiment_info, is_matched, matched_product_id, matched_product_title, category = retrieve_best_response(
        user_input, 
        final_product_query_responses_data,
        prod_df, 
        product_metadata
    )
    
    # Generate chatbot response
    chatbot_reply = generate_chatbot_response(
        user_input, matched_query, matched_response, matched_product_title,
        review_text, sentiment_info, is_matched, matched_product_id, category, prod_df
    )
    
    # Return complete results
    return {
        "user_query": user_input,
        "matched_query": matched_query,
        "suggested_product": matched_product_title,
        "product_category": category,
        "sentiment": sentiment_info,
        "response": chatbot_reply
    }

# %% [markdown]
# ## Customer FAQ'S

# %%
# Define a function to search the index given a user query
def retrieve_faq(query, faq_df, index, model, top_k=1):
    # Compute the query embedding and normalize it
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    
    # Search the index for nearest neighbors
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve FAQ answers
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        faq_question = faq_df.iloc[idx]['utterance']
        faq_answer = faq_df.iloc[idx]['response']
        results.append((faq_question, faq_answer, distance))
    return results

# Function to generate a conversational chatbot response
def general_chatbot_response(user_query, matched_query, matched_response, matched_intent, matched_tone):
    prompt = f"User asked: '{user_query}'. Matched query: '{matched_query}'."
    # Optionally include intent and tone if provided
    if matched_intent:
        prompt += f"Intent: '{matched_intent}'.\n"
    if matched_tone:
        prompt += f"Tone: '{matched_tone}'.\n"
        
    prompt += "Generate a helpful and engaging chatbot response."

    chatbot_response = chatbot(prompt, max_length=100, do_sample=True)[0]["generated_text"]
    return chatbot_response.strip()


# %% [markdown]
# ## Router Function

# %%
def process_user_query():
    query_type = int(input("What query do you have? \n1. Product 2. FAQ\nEnter number:"))
    query = input('Hi! How can I help you?')
    if query_type == 1:
        # Call the function with user input
        recommendation_result = get_product_recommendations(query)
        return recommendation_result["response"]
    else:
        results = retrieve_faq(query,faq_df, index, model)
        return results[0][1]
        
        

# %% [markdown]
# # Testing 

# %%
# Example user input
user_query = "Suggest a phone below 15000 dollars"

# Call the function with user input
recommendation_result = get_product_recommendations(user_query)

# Print the results
print("\n🔹 **User Query:**", recommendation_result["user_query"])
print("📌 **Matched Query:**", recommendation_result["matched_query"])
print("📦 **Suggested Product:**", recommendation_result["suggested_product"])
print("📂 **Product Category:**", recommendation_result["product_category"])
print("📊 **Sentiment Analysis:**", recommendation_result["sentiment"])
print("🤖 **Chatbot Response:**", recommendation_result["response"])


# %%
# Example user input
#user_query = "Suggest a phone below 15000 dollars"
# process_user_query()

# # %%
# #user_query = "You are useless. Get me a human"
# print(process_user_query())

# # %%
# #user_query = "I want a refund"
# print(process_user_query())

# %% [markdown]
# # Chatbot UI

# %%


# Import your functions here. For example:

app = Flask(__name__)

# Home route: serves the UI
@app.route('/')
def home():
    return render_template('index.html')

# Chat route: processes user queries
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    # Extract the query type and query text from the incoming JSON
    query_type = data.get('query_type')  # Expected values: "product" or "faq"
    query = data.get('query')

    # Process based on query type
    if query_type == "product":
        # Process as product-related query
        recommendation_result = get_product_recommendations(query)
        response_text = recommendation_result["response"]
    else:
        # Process as FAQ query
        faq_results = retrieve_faq(query, faq_df, index, model)
        # Assuming you want the answer of the closest FAQ match:
        response_text = faq_results[0][1] if faq_results else "Sorry, I couldn't find an answer to that question."

    return jsonify({'response': response_text})



# %%

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

free_port = get_free_port()
print("Using free port:", free_port)

if __name__ == '__main__':
    app.run(debug=True, port=free_port)


# %%