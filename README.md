# Product Description Generator using Image Captioning Models

## Project Overview

This project aims to automate the generation of product descriptions by leveraging image captioning models. By providing an image of a product, the model will generate a descriptive text that can be used in e-commerce listings, catalogs, or marketing materials. The project consists of data preparation, model training, and final product testing using a sample dataset.

## Directory Structure

The project contains the following key files:

- **Data Preparation.ipynb**: This notebook is used to prepare the dataset for training. It includes data cleaning, preprocessing, and any necessary transformations to make the data suitable for model input.
- **Model Training.ipynb**: This notebook contains the implementation of the image captioning model. It includes model architecture, training loop, loss function, and evaluation metrics. The model is trained on the prepared dataset to generate accurate and relevant product descriptions.
- **finalmenproducts.xlsx**: This Excel file contains the final set of products used for training and testing the model. It includes columns for `id`, `url`, `price`, `description`, and `image`. The `image` column is used as input for the model, and the generated descriptions are compared against the existing `description` column for validation.

## Code Explanation

### 1. **Data Preparation (`Data Preparation.ipynb`)**
   - **Objective:** Prepare image data and corresponding captions for training.
   - **Steps:**
     1. **Loading and Formatting Data:**
        - Load the dataset from an Excel file.
        - Extract relevant columns, such as descriptions and image URLs.
     2. **URL Correction:**
        - Ensure all image URLs are prefixed with `http://`.
     3. **Downloading Images:**
        - Download images from the URLs and save them locally.
     4. **Filtering Data:**
        - Filter the data to include only the images that exist in a specified folder.
     5. **Saving Data:**
        - Save the cleaned and formatted data (captions and local image paths) to a text file.
     6. **Description Cleanup:**
        - Remove specific terms (like "ASOS DESIGN") from the descriptions to prepare captions.

### 2. **Model Training (`model training.ipynb`)**
   - **Objective:** Train a neural network for image captioning using the Inception V3 model as an encoder and an LSTM-based decoder.
   - **Steps:**
     1. **Data Loading and Preprocessing:**
        - Images are loaded and preprocessed.
        - Captions are cleaned, tokenized, and padded.
     2. **Feature Extraction:**
        - Use Inception V3 to extract image features.
     3. **Model Architecture:**
        - A CNN encoder and an RNN decoder are defined.
     4. **Training:**
        - The model is trained over several epochs.
        - The training loop involves computing the loss and updating model weights.
     5. **Caption Generation:**
        - A function generates captions for test images by passing features through the trained model.

## Requirements

To run this project, you need to have the following dependencies installed:

- Python 3.x
- Jupyter Notebook
- TensorFlow
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Tqdm
- Scikit-learn
- NLTK

You can install these dependencies using pip:

```bash
pip install tensorflow numpy pandas opencv-python matplotlib tqdm scikit-learn nltk
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/product-description-generator.git
cd product-description-generator
```

2. Open the Jupyter Notebooks:

```bash
jupyter notebook
```

3. Start with the `Data Preparation.ipynb` to prepare your dataset.
4. Proceed to `Model Training.ipynb` to train the image captioning model.
5. Test the model using the data in `finalmenproducts.xlsx`.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

