import pandas as pd
import unicodedata
import os
import numpy as np

ALLERGENS_KEYWORDS = {
    'Dairy': [
        'leche', 'queso', 'mantequilla', 'yogur', 'crema', 'lactosa', 'caseina',
        'milk', 'cheese', 'butter', 'yogurt', 'cream', 'lactose', 'casein', 'dairy'
    ],
    'Gluten': [
        'trigo', 'harina', 'pan', 'fideos', 'pasta', 'cebada', 'centeno', 'avena', 'semola',
        'wheat', 'flour', 'bread', 'pasta', 'barley', 'rye', 'oats', 'semolina', 'gluten'
    ],
    'Peanuts': [
        'maní', 'mani', 'cacahuete', 'mantequilla de maní',
        'peanut', 'peanut butter'
    ],
    'Seafood': [
        'camarón', 'camaron', 'langostino', 'cangrejo', 'choros', 'conchas', 'pulpo', 'calamar', 'marisco', 'pescado',
        'shrimp', 'prawn', 'crab', 'mussels', 'clams', 'octopus', 'squid', 'seafood', 'fish', 'salmon', 'tuna'
    ],
    'Eggs': [
        'huevo', 'yema', 'clara', 'mayonesa',
        'egg', 'yolk', 'albumin', 'mayonnaise'
    ],
    'Soy': [
        'soja', 'soya', 'tofu', 'sillao', 'salsa de soja', 'edamame',
        'soy', 'tofu', 'soy sauce', 'edamame'
    ],
    'Tree_Nuts': [
        'nuez', 'pecana', 'almendra', 'castaña', 'avellana', 'pistacho', 'marañon',
        'walnut', 'pecan', 'almond', 'cashew', 'hazelnut', 'pistachio', 'brazil nut'
    ]
}


DATASET_CONFIGS = {
    'recetasdelaabuela': {
        'file': 'datasets/recetasdelaabuela.csv',
        'encoding': 'utf-8',
        'columns': {
            'Nombre': 'recipe_name',
            'Ingredientes': 'ingredients_text',
            'Pasos': 'instructions'
        },
        'language': 'es',
        'source': 'Recetas de la Abuela'
    },
    'recetamx': {
        'file': 'datasets/recetamx.csv',
        'encoding': 'utf-8',
        'columns': {
            'Nombre': 'recipe_name',
            'Ingredientes': 'ingredients_text',
            'Pasos': 'instructions'
        },
        'language': 'es',
        'source': 'Recetas MX'
    }
}

def normalize_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    text = str(text).strip()
    if text.lower() in ['nan', 'null', 'none', '[]', '{}', 'n/a']:
        return ""
    
    return ''.join(
        c for c in unicodedata.normalize('NFD', text.lower())
        if unicodedata.category(c) != 'Mn'
    )

def tag_recipe_allergens(ingredients_text, allergens_dict):
    normalized_ingredients = normalize_text(ingredients_text)
    
    if not normalized_ingredients:
        found_allergens = {}
        for allergen_name in allergens_dict.keys():
            found_allergens[f'contains_{allergen_name}'] = False
        return found_allergens
    
    found_allergens = {}
    for allergen_name, keywords in allergens_dict.items():
        is_present = False
        for keyword in keywords:
            normalized_keyword = normalize_text(keyword)
            if normalized_keyword and (
                f' {normalized_keyword} ' in f' {normalized_ingredients} ' or
                normalized_ingredients.startswith(normalized_keyword + ' ') or
                normalized_ingredients.endswith(' ' + normalized_keyword) or
                normalized_ingredients == normalized_keyword
            ):
                is_present = True
                break
        found_allergens[f'contains_{allergen_name}'] = is_present
        
    return found_allergens

def clean_and_validate_data(df, source_name):
    print(f"  Limpiando datos de {source_name}...")
    initial_count = len(df)
    
    null_values = ['nan', 'NaN', 'null', 'NULL', 'None', 'NONE', '', '[]', '{}', 'No disponible', 'n/a', 'N/A']
    df = df.replace(null_values, np.nan)
    
    df = df.dropna(subset=['recipe_name', 'ingredients_text'])
    
    df['recipe_name'] = df['recipe_name'].astype(str).str.strip()
    df['ingredients_text'] = df['ingredients_text'].astype(str).str.strip()
    df['instructions'] = df['instructions'].astype(str).str.strip()
    
    df = df[df['recipe_name'].str.len() > 0]
    df = df[df['ingredients_text'].str.len() > 0]
    
    df['instructions'] = df['instructions'].replace(['nan', 'NaN', '[]'], 'Sin instrucciones disponibles')
    
    df = df.drop_duplicates(subset=['recipe_name'], keep='first')
    
    df = df.reset_index(drop=True)
    
    final_count = len(df)
    print(f"    {source_name}: {initial_count} → {final_count} recetas ({final_count/initial_count*100:.1f}% conservadas)")
    
    return df

def process_single_dataset(dataset_name, config):
    print(f"\n Processing dataset: {dataset_name}")
    
    file_path = config['file']
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    df = None
    for encoding in [config['encoding'], 'utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Loaded with encoding: {encoding}")
            break
        except Exception as e:
            continue
    
    if df is None:
        print(f"Could not load the file")
        return None
    
    print(f"Original dimensions: {df.shape}")
    
    required_cols = list(config['columns'].keys())
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return None
    
    df_processed = df[required_cols].copy()
    df_processed.rename(columns=config['columns'], inplace=True)

    df_processed['language'] = config['language']
    df_processed['source'] = config['source']
    df_processed['dataset_name'] = dataset_name
    
    df_processed = clean_and_validate_data(df_processed, config['source'])
    
    if len(df_processed) == 0:
        print(f"No valid data after cleaning")
        return None
    
    print(f"Detecting allergens...")
    allergen_results = df_processed['ingredients_text'].apply(
        lambda ingredients: tag_recipe_allergens(ingredients, ALLERGENS_KEYWORDS)
    )
    
    df_allergens = pd.json_normalize(allergen_results)
    df_final = pd.concat([df_processed, df_allergens], axis=1)
    
    print(f"Processing completed: {len(df_final)} recipes")
    
    return df_final

def combine_all_datasets(dataset_configs, output_file='processed_recipes.csv'):
    print("Starting multi-dataset processing")
    print("=" * 60)
    
    all_datasets = []
    processing_stats = {}
    
    for dataset_name, config in dataset_configs.items():
        try:
            df_processed = process_single_dataset(dataset_name, config)
            if df_processed is not None:
                all_datasets.append(df_processed)
                processing_stats[dataset_name] = {
                    'recipes': len(df_processed),
                    'language': config['language'],
                    'source': config['source']
                }
            else:
                processing_stats[dataset_name] = {'error': 'Failed to process'}
        except Exception as e:
            print(f"Error procesando {dataset_name}: {e}")
            processing_stats[dataset_name] = {'error': str(e)}
    
    if not all_datasets:
        print("No datasets were successfully processed")
        return
    
    print(f"\nCombining {len(all_datasets)} datasets...")
    combined_df = pd.concat(all_datasets, ignore_index=True)
    
    initial_combined = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['recipe_name'], keep='first')
    final_combined = len(combined_df)
    
    print(f"Duplicates removed: {initial_combined - final_combined}")
    
    combined_df.to_csv(output_file, index=False, encoding='utf-8')

    print("\n" + "=" * 60)
    print("Overview")
    print("=" * 60)
    
    print(f"Final file: {output_file}")
    print(f"Total recipes: {len(combined_df)}")
    
    print("\nBy dataset:")
    for dataset_name, stats in processing_stats.items():
        if 'error' in stats:
            print(f"{dataset_name}: {stats['error']}")
        else:
            print(f" {dataset_name}: {stats['recipes']} recetas ({stats['language']}) - {stats['source']}")
    
    print("\n By language:")
    language_counts = combined_df['language'].value_counts()
    for lang, count in language_counts.items():
        percentage = (count / len(combined_df)) * 100
        print(f"  {lang}: {count} recetas ({percentage:.1f}%)")
    
    print("\n Allergen statistics:")
    allergen_cols = [col for col in combined_df.columns if col.startswith('contains_')]
    for col in allergen_cols:
        count = combined_df[col].sum()
        percentage = (count / len(combined_df)) * 100
        allergen_name = col.replace('contains_', '')
        print(f"  {allergen_name}: {count} recetas ({percentage:.1f}%)")
    
    return combined_df

def add_new_dataset(dataset_name, file_path, column_mapping, language='en', source='Unknown'):
    DATASET_CONFIGS[dataset_name] = {
        'file': file_path,
        'encoding': 'utf-8',
        'columns': column_mapping,
        'language': language,
        'source': source
    }
    print(f"Dataset '{dataset_name}' added to the configuration")

if __name__ == "__main__":
    combined_df = combine_all_datasets(DATASET_CONFIGS)