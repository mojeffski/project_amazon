import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(notebook_path, output_path=None):
    with open(notebook_path) as f:
        notebook = nbformat.read(f, as_version=4)
        
    # Set up the notebook execution
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    # Execute the notebook
    ep.preprocess(notebook, {'metadata': {'path': './'}})
    
    # Optionally, save the executed notebook
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(notebook, f)

if __name__ == "__main__":
    # Run the first notebook:
    run_notebook('1_sentiment_classifier.ipynb', '1_sentiment_classifier_results.ipynb')
    
    # Run the second notebook:
    run_notebook('2_topic_clustering.ipynb.ipynb', 'results/2_topic_clustering.ipynb_results.ipynb')
    # Run the third notebook:
    run_notebook('3_genai_blogposts.ipynb', 'results/3_genai_blogposts_results.ipynb')
    # Run the second notebook