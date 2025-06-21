import pandas as pd
import unicodedata
import os
import numpy as np

ALLERGENS_KEYWORDS = {
    'Dairy': ['leche', 'queso', 'mantequilla', 'yogur', 'crema', 'lactosa', 'caseina', 'milk', 'cheese', 'butter', 'yogurt', 'cream', 'lactose', 'casein'],
    'Gluten': ['trigo', 'harina', 'pan', 'fideos', 'pasta', 'cebada', 'centeno', 'avena', 'semola', 'wheat', 'flour', 'bread', 'pasta', 'barley', 'rye', 'oats', 'semolina'],
    'Peanuts': ['maní', 'mani', 'cacahuete', 'mantequilla de maní', 'peanut', 'peanut butter'],
    'Seafood': ['camarón', 'camaron', 'langostino', 'cangrejo', 'choros', 'conchas', 'pulpo', 'calamar', 'marisco', 'pescado', 'shrimp', 'prawn', 'crab', 'mussels', 'clams', 'octopus', 'squid', 'seafood', 'fish'],
    'Eggs': ['huevo', 'yema', 'clara', 'mayonesa', 'egg', 'yolk', 'albumin', 'mayonnaise'],
    'Soy': ['soja', 'soya', 'tofu', 'sillao', 'salsa de soja', 'edamame', 'soy', 'tofu', 'soy sauce'],
    'Tree_Nuts': ['nuez', 'pecana', 'almendra', 'castaña', 'avellana', 'pistacho', 'marañon', 'castana', 'walnut', 'pecan', 'almond', 'cashew', 'hazelnut', 'pistachio']
}

INPUT_FILENAME = 'recetasdelaabuela.csv'
OUTPUT_FILENAME = 'processed_recipes.csv'

def normalize_text(text):
    """
    Normaliza un string convirtiéndolo a minúsculas y removiendo acentos.
    Retorna string vacío si el input no es string (ej. NaN).
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Limpiar texto extra
    text = str(text).strip()
    if text.lower() in ['nan', 'null', 'none', '[]', '{}']:
        return ""
    
    # NFD normalization separa caracteres y sus acentos,
    # luego removemos los acentos (categoría Mn).
    return ''.join(
        c for c in unicodedata.normalize('NFD', text.lower())
        if unicodedata.category(c) != 'Mn'
    )

def tag_recipe_allergens(ingredients_text, allergens_dict):
    """
    Revisa un string de ingredientes por alérgenos y retorna un diccionario de flags booleanos.
    """
    normalized_ingredients = normalize_text(ingredients_text)
    
    # Si no hay ingredientes válidos, retornar todo False
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
            # Revisamos la palabra completa para evitar matches parciales
            if f' {normalized_keyword} ' in f' {normalized_ingredients} ' or \
               normalized_ingredients.startswith(normalized_keyword + ' ') or \
               normalized_ingredients.endswith(' ' + normalized_keyword) or \
               normalized_ingredients == normalized_keyword:
                is_present = True
                break
        found_allergens[f'contains_{allergen_name}'] = is_present
        
    return found_allergens

def clean_and_validate_data(df):
    """
    Limpia y valida los datos del DataFrame
    """
    print("Iniciando limpieza de datos...")
    
    # Mostrar estadísticas iniciales
    print(f"Filas iniciales: {len(df)}")
    
    # Limpiar valores que representan datos faltantes
    null_values = ['nan', 'NaN', 'null', 'NULL', 'None', 'NONE', '', '[]', '{}', 'No disponible']
    df = df.replace(null_values, np.nan)
    
    # Mostrar información sobre valores nulos
    print("\nValores nulos por columna:")
    print(df.isnull().sum())
    
    # Eliminar filas donde el nombre de la receta esté vacío
    df = df.dropna(subset=['recipe_name'])
    print(f"Después de eliminar recetas sin nombre: {len(df)} filas")
    
    # Eliminar filas donde los ingredientes estén vacíos
    df = df.dropna(subset=['ingredients_text'])
    print(f"Después de eliminar recetas sin ingredientes: {len(df)} filas")
    
    # Limpiar strings vacíos adicionales
    df['recipe_name'] = df['recipe_name'].astype(str).str.strip()
    df['ingredients_text'] = df['ingredients_text'].astype(str).str.strip()
    df['instructions'] = df['instructions'].astype(str).str.strip()
    
    # Eliminar filas donde recipe_name o ingredients_text sean solo whitespace
    df = df[df['recipe_name'].str.len() > 0]
    df = df[df['ingredients_text'].str.len() > 0]
    print(f"Después de eliminar strings vacíos: {len(df)} filas")
    
    # Reemplazar 'nan' strings con valores más descriptivos para instrucciones
    df['instructions'] = df['instructions'].replace(['nan', 'NaN', '[]'], 'Sin instrucciones')
    
    # Reset index después de la limpieza
    df = df.reset_index(drop=True)
    
    print(f"Datos limpiados. Filas finales: {len(df)}")
    return df

def process_recipe_data(input_file, output_file):
    """
    Función principal para cargar, procesar, etiquetar y guardar los datos de recetas.
    """
    # Verificar si el archivo de entrada existe
    if not os.path.exists(input_file):
        print(f"Error: El archivo de entrada '{input_file}' no fue encontrado.")
        return

    print("Iniciando el preprocesamiento de datos...")
    
    # Cargar el dataset
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(input_file, encoding='latin-1')
        except:
            df = pd.read_csv(input_file, encoding='cp1252')
    
    print(f"Dataset cargado exitosamente con {df.shape[0]} filas y {df.shape[1]} columnas.")
    print(f"Columnas encontradas: {df.columns.tolist()}")

    # Mapeo de columnas basado en tu dataset
    column_mapping = {
        'Nombre': 'recipe_name',
        'Ingredientes': 'ingredients_text', 
        'Pasos': 'instructions'
    }
    
    # Verificar si las columnas requeridas existen
    required_cols = list(column_mapping.keys())
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Error: Faltan columnas requeridas: {missing_cols}")
        print(f"Columnas encontradas en el archivo: {df.columns.tolist()}")
        return

    # Seleccionar y renombrar columnas
    df_processed = df[required_cols].copy()
    df_processed.rename(columns=column_mapping, inplace=True)
    
    print("Columnas estandarizadas exitosamente.")
    
    # Limpiar y validar datos
    df_processed = clean_and_validate_data(df_processed)
    
    if len(df_processed) == 0:
        print("Error: No quedan datos válidos después de la limpieza.")
        return
    
    print("Iniciando etiquetado de alérgenos...")

    # Aplicar la función de etiquetado de alérgenos a cada fila
    allergen_results = df_processed['ingredients_text'].apply(
        lambda ingredients: tag_recipe_allergens(ingredients, ALLERGENS_KEYWORDS)
    )

    # Convertir la Serie de diccionarios en un DataFrame
    df_allergens = pd.json_normalize(allergen_results)

    # Concatenar los datos originales procesados con las nuevas flags de alérgenos
    df_final = pd.concat([df_processed, df_allergens], axis=1)
    
    print("Etiquetado de alérgenos completado.")

    # Guardar el DataFrame final en un nuevo archivo CSV
    df_final.to_csv(output_file, index=False, encoding='utf-8')
    
    print("-" * 60)
    print(f"¡Procesamiento finalizado exitosamente!")
    print(f"El dataset normalizado ha sido guardado como: '{output_file}'")
    print(f"Recetas procesadas: {len(df_final)}")
    print("-" * 60)
    
    # Mostrar estadísticas finales
    print("\nEstadísticas de alérgenos:")
    allergen_cols = [col for col in df_final.columns if col.startswith('contains_')]
    for col in allergen_cols:
        count = df_final[col].sum()
        percentage = (count / len(df_final)) * 100
        allergen_name = col.replace('contains_', '')
        print(f"  {allergen_name}: {count} recetas ({percentage:.1f}%)")
    
    print(f"\nVista previa de las primeras 5 recetas:")
    print(df_final[['recipe_name', 'ingredients_text']].head())
    
    print(f"\nColumnas del dataset final:")
    print(df_final.columns.tolist())

if __name__ == "__main__":
    process_recipe_data(INPUT_FILENAME, OUTPUT_FILENAME)