# AI-Generated Product Review Website

This open-source project creates a Flask website that displays AI-generated product reviews based on an original dataset of Amazon product reviews. The reviews are generated using a combination of sentiment analysis, topic clustering, and natural language generation (NLG) models. The website also includes an evaluation of the models used and provides insights into how the reviews were created.

## Features

- AI-powered sentiment classification on Amazon product reviews
- Topic modeling and clustering to categorize product reviews
- Generated blog posts for each product, showcasing model capabilities
- Flask web app to display generated reviews and model evaluations

## Tech Stack

- **Python**: 3.12.7
- **Flask**: For serving the web app
- **Hugging Face Transformers**: For sentiment classification and text generation
- **Gensim**: For topic modeling (LDA)
- **SQLite**: For storing and querying generated blog posts and model evaluations

## Installation Guide

### Prerequisites
- **Python 3.12.7** installed
- **Git** for cloning the repository
- **Hugging Face API token** (you need this for model access)

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/your-repository-name.git
   cd your-repository-name

## Install the required dependencies

```bash
pip install -r requirements.txt

1. *Create an .env file in the project root folder and add your Hugging Face token:**
    HUGGINGFACE_API_KEY=your-huggingface-token

## Create a .gitignore file to prevent certain files from being committed (recommended):

```bash
# Add the following to your .gitignore
venv/
flask_project/data/blogposts.db
__pycache__/
.env
datasets/
fine_tuned_model/
models/
*.ipynb_checkpoints