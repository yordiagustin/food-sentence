# Food Sentence - Safe Recipe Recommender

A deep learning-powered recipe recommendation system that helps people with food allergies find safe recipes. The system uses semantic search and allergen detection to provide personalized recipe recommendations.

## 🚀 Features

- **Allergen Detection**: Automatically detects common allergens in recipe ingredients
- **Semantic Search**: Uses multilingual embeddings to find recipes based on natural language queries
- **Safety First**: Filters out recipes containing allergens that users need to avoid
- **Multi-language Support**: Works with Spanish and English recipes
- **RESTful API**: Easy-to-use API with Swagger documentation

## 📁 Project Structure

```
food-sentence/
├── datasets/                    # Place your recipe datasets here
│   ├── recetasdelaabuela.csv   # Example dataset
│   └── recetamx.csv           # Example dataset
├── data_cleaning.py            # Data preprocessing and cleaning
├── generate_embeddings.py      # Generate recipe embeddings
├── api.py                      # FastAPI application
├── main.py                     # Alternative API entry point
├── processed_recipes.csv       # Cleaned and processed recipes
├── recipe_embeddings.npy       # Recipe embeddings
└── README.md                   # This file
```

## 🛠️ Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Datasets

Place your recipe datasets in the `datasets/` folder. The system expects CSV files with the following column structure:

**Required Columns:**
- `Nombre` (or `recipe_name`): Recipe name
- `Ingredientes` (or `ingredients_text`): List of ingredients
- `Pasos` (or `instructions`): Cooking instructions

**Example dataset format:**
```csv
Nombre,Ingredientes,Pasos
"Pasta Carbonara","pasta, huevos, queso parmesano, panceta","1. Cocinar pasta..."
"Ensalada César","lechuga, crutones, parmesano, aderezo","1. Lavar lechuga..."
```

### 3. Configure Dataset Mapping

If your datasets have different column names, you can modify the `DATASET_CONFIGS` in `data_cleaning.py`:

```python
DATASET_CONFIGS = {
    'your_dataset_name': {
        'file': 'datasets/your_dataset.csv',
        'encoding': 'utf-8',
        'columns': {
            'Your_Recipe_Name_Column': 'recipe_name',
            'Your_Ingredients_Column': 'ingredients_text',
            'Your_Instructions_Column': 'instructions'
        },
        'language': 'es',  # or 'en'
        'source': 'Your Dataset Source'
    }
}
```

### 4. Run Data Processing Pipeline

#### Step 1: Clean and Process Data
```bash
python data_cleaning.py
```

This will:
- Load all datasets from the `datasets/` folder
- Clean and validate the data
- Detect allergens in ingredients
- Combine all datasets into `processed_recipes.csv`

#### Step 2: Generate Embeddings
```bash
python generate_embeddings.py
```

This will:
- Load the processed recipes
- Generate semantic embeddings using a multilingual model
- Save embeddings to `recipe_embeddings.npy`

### 5. Start the API

```bash
python api.py
```

The API will be available at:
- **Main API**: http://127.0.0.1:8000
- **Swagger Documentation**: http://127.0.0.1:8000/docs
- **ReDoc Documentation**: http://127.0.0.1:8000/redoc

## 🍽️ API Usage

### Get Recipe Recommendations

**Endpoint:** `POST /recommend`

**Request Body:**
```json
{
  "user_query": "algo cremoso para cenar",
  "allergies": ["Dairy", "Nuts"],
  "top_n": 5
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "recipe_name": "Pasta con Salsa de Tomate",
      "similarity_score": 0.85,
      "ingredients_text": "pasta, tomates, aceite de oliva...",
      "instructions": "1. Cocinar pasta...",
      "source": "Recetas de la Abuela",
      "language": "es"
    }
  ],
  "metrics": {
    "total_time_ms": 150.2,
    "safe_recipes_after_filtering": 1250,
    "recipes_returned": 5
  }
}
```

### Supported Allergens

The system automatically detects these allergens:
- `Dairy` (leche, queso, mantequilla, etc.)
- `Gluten` (trigo, harina, pan, etc.)
- `Peanuts` (maní, cacahuete, etc.)
- `Seafood` (camarón, pescado, etc.)
- `Eggs` (huevo, yema, etc.)
- `Soy` (soja, tofu, etc.)
- `Tree_Nuts` (nuez, almendra, etc.)

## 🔧 Configuration

### Model Configuration

The system uses the `paraphrase-multilingual-MiniLM-L12-v2` model for generating embeddings. You can modify the model in `generate_embeddings.py` and `api.py`.

### Allergen Keywords

You can customize allergen detection by modifying the `ALLERGENS_KEYWORDS` dictionary in `data_cleaning.py`.

## 📊 Data Processing Details

### Data Cleaning Steps

1. **Null Value Handling**: Removes recipes with missing names or ingredients
2. **Text Normalization**: Removes accents and normalizes text
3. **Duplicate Removal**: Removes duplicate recipes based on name
4. **Allergen Detection**: Scans ingredients for allergen keywords
5. **Data Validation**: Ensures minimum text length requirements

### Embedding Generation

- Uses multilingual sentence transformers
- Combines recipe name, ingredients, and instructions
- Generates 384-dimensional embeddings
- Optimized for Spanish and English content

## 🚨 Safety Features

- **Mandatory Allergy Declaration**: Users must specify allergies for safety
- **Automatic Filtering**: Recipes containing allergens are automatically excluded
- **Clear Warnings**: API provides warnings when allergen columns are missing
- **Audit Trail**: All filtering decisions are logged

## 🐛 Troubleshooting

### Common Issues

1. **File Not Found Errors**
   - Ensure datasets are in the `datasets/` folder
   - Check file paths in `DATASET_CONFIGS`

2. **Encoding Issues**
   - The system tries multiple encodings (utf-8, latin-1, cp1252)
   - Check your CSV file encoding

3. **Memory Issues**
   - Large datasets may require more RAM
   - Consider processing datasets in smaller batches

4. **API Not Starting**
   - Ensure all required files exist (`processed_recipes.csv`, `recipe_embeddings.npy`)
   - Check port 8000 is available

### Logs and Debugging

The API provides detailed logging:
- Data loading status
- Allergen filtering decisions
- Processing times
- Error messages

## 📈 Performance

- **Typical Response Time**: 100-200ms
- **Supported Languages**: Spanish, English
- **Recommended Dataset Size**: Up to 100,000 recipes
- **Memory Usage**: ~500MB for 50,000 recipes

## Evaluate Embeddings
To evaluate embeddings you can run main.py file. It evaluates only the similarity directly with the embeddings.


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your datasets to the `datasets/` folder
4. Update the `DATASET_CONFIGS` if needed
5. Test the pipeline
6. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation at http://127.0.0.1:8000/docs
3. Open an issue on GitHub