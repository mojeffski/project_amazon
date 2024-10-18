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

2. **Install the required dependencies**

    ```bash
    pip install -r requirements.txt
    ```
3. **Create an .env file in the project root folder and add your Hugging Face token:**
    HUGGINGFACE_API_KEY=your-huggingface-token

4. **Create a .gitignore file to prevent certain files from being committed (recommended):**

```bash
# Add the following to your .gitignore
- venv/
- flask_project/data/blogposts.db
- __pycache__/
- .env
- datasets/
- fine_tuned_model/
- models/
- *.ipynb_checkpoints
```

### How to Run

5. **Run the main script:**
    ```bash
    python main.py
    ```
This will execute three Jupyter notebooks consecutively:

1. **1_sentiment_classifier.ipynb:**
- Outputs cleaned_with_sentiment_numeric.csv, which contains a new column sentiment with model-generated sentiments. Saves evaluation metrics in blogposts.db for the methods.html page.
2. **2_topic_clustering.ipynb:**
- Uses LDA to create topics for the product categories. Outputs amazon_review_categories.csv in dataset/interim and lda_visualization.html in the templates folder for visualization in the Flask app.
3. **3_genai_blogposts.ipynb:**
- Generates blog posts based on product names and their corresponding summaries. Saves the results in the blogposts table inside blogposts.db.

### Run the Flask app:
Navigate to the flask_project folder and run the Flask development server::

```bash
cd flask_project
python app.py
```
- This will launch the Flask web app on 
    ```bash
    http://127.0.0.1:5000/.
    ```

### Project Structure
    ```bash
        ├── datasets/
        │   ├── interim/         # Holds intermediate datasets such as 'amazon_review_categories.csv'
        │   └── raw/             # Raw datasets
        ├── flask_project/
        │   ├── data/            # SQLite database for blog posts
        │   ├── static/          # Static assets (CSS, JS, images)
        │   ├── templates/       # HTML templates for Flask app
        │   └── app.py           # Flask application file
        ├── models/              # Fine-tuned models for sentiment analysis
        ├── fine_tuned_model/    # Fine-tuned model weights
        ├── 1_sentiment_classifier.ipynb
        ├── 2_topic_clustering.ipynb
        ├── 3_genai_blogposts.ipynb
        └── main.py              # Script to run all notebooks
    ```
### Contributing
- Feel free to open issues, submit pull requests, and contribute to this open-source project. Please follow the contribution guidelines.

### License
- This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgments
- Hugging Face for the pre-trained models
- Gensim for topic modeling tools
- Flask for the web framework