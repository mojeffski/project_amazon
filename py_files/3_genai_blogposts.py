#!/usr/bin/env python
# coding: utf-8

# ### 3. GenAi-task

import sqlite3
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import pipeline
import torch
import pandas as pd
import os
import random
from dotenv import load_dotenv

from huggingface_hub import login

# Load environment variables from the .env file
load_dotenv()

# Access the Hugging Face token
hf_token = os.getenv("HF_TOKEN")

# Use the token to authenticate with Hugging Face
login(token=hf_token)

project_path = "dataset/interim/cleaned_with_sentiment_numeric.csv"
categories_path = "dataset/interim/amazon_review_categories.csv"



df = pd.read_csv(project_path)


df_categories = pd.read_csv(categories_path)


df_categories.head()

# clean names:
# Clean product names: remove special characters, extra commas, and strip
# whitespace
df["name"] = (
    df["name"].replace({r"\r": " ", r"\n": " ", ",,": ","}, regex=True).str.strip()
)

# Remove any remaining multiple spaces
df["name"] = df["name"].replace(r"\s+", " ", regex=True)

df["name_clean"] = df["name"].str.split(",").str[0]

df_all = pd.merge(
    df,
    df_categories[["category", "label"]],
    how="left",
    left_on="categories",
    right_on="category",
)

grouped_reviews = (
    df_all.groupby(["label", "name_clean"])["reviews.text"].apply(list).reset_index()
)

# Assume the column with lists is named 'list_column'
# Create a new column 'list_length' that stores the number of items in
# each list
grouped_reviews["nr_reviews"] = grouped_reviews["reviews.text"].apply(len)

ratings_helper = pd.DataFrame(
    df_all.groupby("name_clean")["sentiment"].mean()
).reset_index()

df_final = pd.merge(grouped_reviews, ratings_helper, on="name_clean")

df_final_filtered = df_final[df_final["nr_reviews"] >= 5]

df_final_filtered.head(2)

df_sorted = df_final_filtered.sort_values(
    by=["label", "sentiment"], ascending=[True, False]
)

# Group by 'category' and select the top 3 rows per group
top_3_per_category = df_sorted.groupby("label").head(3)
last_3_per_category = df_sorted.groupby("label").tail(3)

top_3_per_category["cat"] = "5 Star"
last_3_per_category["cat"] = "Fiasko"

top_categories = pd.concat([top_3_per_category, last_3_per_category])
top_categories = top_categories.drop_duplicates(subset=["name_clean"])

# Function to randomly select 9 reviews from the list of reviews
def select_random_reviews(reviews_list):
    # If there are fewer than 3 reviews, return them all
    if len(reviews_list) <= 20:
        return reviews_list
    # Otherwise, return 3 random reviews
    return random.sample(reviews_list, 20)

top_categories["reviews.text"] = top_categories["reviews.text"].apply(
    select_random_reviews
)

for f in top_categories["reviews.text"]:
    print(f)


# ## GenAi
#
# #### SUMMARIZATION
# Function to randomly select  reviews and generate a summary
def summarize_reviews(reviews_list, num_reviews=15):
    # If there are fewer reviews than num_reviews, use all reviews
    if len(reviews_list) < num_reviews:
        selected_reviews = reviews_list
    else:
        # Randomly select 5 reviews from the list
        selected_reviews = random.sample(reviews_list, num_reviews)

    # Convert the selected reviews to a single string
    reviews_text = " ".join(selected_reviews)

    # Generate a summary
    summary = summarizer(reviews_text, max_length=150, min_length=50, do_sample=False)

    # Return the summary text
    return summary[0]["summary_text"]

device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)

# Load the summarization pipeline (you can use BART or another model)
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=0 if torch.backends.mps.is_available() else -1,
)

top_categories["summary"] = top_categories["reviews.text"].apply(summarize_reviews)

# Save the DataFrame to a new CSV
top_categories.to_csv("dataset/interim/products_with_summaries.csv", index=False)

# Display the updated DataFrame with summaries
top_categories.head()


# ### Cerate Posts:
# # Generate blogposts blogposts:
# Load the tokenizer and model
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to the correct device

# Initialize the pipeline for text generation
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device.type == "cuda" else -1,
)


def generate_blog_post(product_name, summary):
    prompt = f"Write a blog post about the product '{product_name}' based on the following review summary: {summary}"

    inputs = tokenizer(prompt, return_tensors="pt").to(
        device
    )  # Move inputs to the correct device
    input_ids = inputs["input_ids"]

    # Generate text with attention_mask and max_length control
    outputs = model.generate(
        input_ids,
        # Add attention_mask to inputs
        attention_mask=inputs["attention_mask"],
        max_length=500,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,  # Set pad_token_id explicitly
    )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Split the text at the first two new lines (\n\n) and keep the part after
    # the split
    # Ensure you only keep the blog post text
    blog_post = generated_text.split("\n\n", 1)[-1].strip()

    return blog_post

# Add a new column to store the generated blog posts
top_categories["blogpost"] = ""

# Loop over each row in the DataFrame and generate the blog post
for idx, row in top_categories.iterrows():

    product_name = row["name_clean"]
    summary = row["summary"]
    cat = row["cat"]
    print(product_name)
    print(cat)
    print(summary)
    # Generate the blog post for the current product
    blog_post = generate_blog_post(product_name, summary)

    # Store the generated blog post in the new 'blogpost' column for the
    # current row
    top_categories.at[idx, "blogpost"] = blog_post

# Print a sample blog post for a category
for idx, row in top_categories.iterrows():
    print(
        f"Blog post for product '{
            row['name_clean']}' in category '{
            row['label']}':\n{
                row['blogpost']}\n"
    )
for idx, row in top_categories.iterrows():
    print(
        f"Blog post for product '{
            row['name_clean']}' in category '{
            row['label']}':\n{
                row['blogpost']}\n"
    )

# #### export results
# Connect to the SQLite database
db_path = "flask_project/data/blogposts.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Drop the 'blogposts' table if it exists
cursor.execute(
    """
DROP TABLE IF EXISTS blogposts
"""
)

# Create the 'blogposts' table with columns based on the DataFrame,
# including 'id' as the primary key
cursor.execute(
    """
CREATE TABLE blogposts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT,
    name_clean TEXT,
    reviews_text TEXT,
    nr_reviews INTEGER,
    sentiment REAL,
    cat TEXT,
    summary TEXT,
    blogpost TEXT
)
"""
)

# Commit the table creation
conn.commit()

# Insert data from the DataFrame into the 'blogposts' table
for idx, row in top_categories.iterrows():
    cursor.execute(
        """
        INSERT INTO blogposts (label, name_clean, reviews_text, nr_reviews, sentiment, cat, summary, blogpost)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            row["label"],
            row["name_clean"],
            ", ".join(row["reviews.text"]),
            row["nr_reviews"],
            row["sentiment"],
            row["cat"],
            row["summary"],
            row["blogpost"],
        ),
    )

# Commit the inserts
conn.commit()

# Close the connection
conn.close()

print("Data successfully saved to the 'blogposts' table with 'id' in blogposts.db.")
