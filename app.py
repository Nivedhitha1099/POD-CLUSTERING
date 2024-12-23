import os
import shutil
import uuid
import json
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np
import zipfile
import logging
import time


logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
parent_stack = []
UPLOAD_FOLDER = 'uploads'
CLUSTER_OUTPUT_FOLDER = 'clustered_output'
TEMP_FOLDER = 'extract'
PATTERN_FILE_PATH = 'pattern_final12.json'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLUSTER_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'zip', 'epub'}
tags = [
    'br', 'h5', 'a', 'figcaption', 'ul', 'img', 'tr', 'caption', 'section', 'aside', 'header',
    'figure', 'i', 'th', 'div', 'span', 'h4', 'h2', 'table', 'h3', 'sup', 'em', 'p', 'strong',
    'td', 'tbody', 'ol', 'h1', 'li', 'colgroup', 'dfn', 'thead', 'title', 'col'
]
levels = range(1, 14)
decorative_tags = ['em', 'strong', 'i', 'br', 'span', 'dfn']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_attributes(tag):
    if tag:
        return ", ".join([f'{attr}="{val}"' for attr, val in tag.attrs.items()])
    return ""


def extract_fragments(html_content, initial_fragment_id, fragment_name, initial_parent_tag):
    global parent_stack
    parent_stack = []
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        data = []
        fragment_id = initial_fragment_id
        for section in soup.find_all('section'):
            fragment_id += 1
            parent_tag = initial_parent_tag
            for child in section.children:
                if child.name and child.name not in decorative_tags:
                    parse_element(child, 1, str(uuid.uuid4()), data, fragment_name, parent_tag)
        return data, fragment_id
    
    except Exception as e:
        print(f"Error processing fragment {fragment_id}: {e}")
        return [], initial_fragment_id

def parse_element(element, level, fragment_id, data, fragment_name, parent_tag):
    presence_encoding = {f'Level_{lvl}_{tag}': 0 for lvl in levels for tag in tags}
    presence_encoding[f'Level_{level}_{element.name}'] = 1
    element_type = element.name
    element_class = " ".join(element.get('class', []))
    attributes = extract_attributes(element)
    fragment_data = {
        'fragment_id': fragment_id,
        'fragment_name': fragment_name,
        'Level': level,
        'element_type': element_type,
        'element_class': element_class,
        'attributes': attributes,
        **presence_encoding
    }
    data.append(fragment_data)
    
    # Process children only after pushing the current element onto the stack
    for child in element.children:
        if child.name and child.name not in decorative_tags:
            new_level = level + 1
            parse_child_element(child, new_level, fragment_id, data, fragment_name, element.name)

def parse_child_element(element, level, fragment_id, data, fragment_name, parent_tag):
    presence_encoding = data[-1]
    presence_encoding[f'Level_{level}_{element.name}'] = 1
    presence_encoding['attributes'] += "; " + extract_attributes(element)
    
    # Process children
    for child in element.children:
        if child.name and child.name not in decorative_tags:
            new_level = level + 1
            parse_child_element(child, new_level, fragment_id, data, fragment_name, element.name)

def perform_clustering(df):
    presence_encoding_cols = [col for col in df.columns if col.startswith("Level_")]
    scaler = StandardScaler()
    df_normalized = scaler.fit_transform(df[presence_encoding_cols])
    dbscan = DBSCAN(eps=0.8, min_samples=3)
    df['cluster'] = dbscan.fit_predict(df_normalized)
    return df

def compare_fragment(row, pattern_data, encoding_cols_combined, encoding_cols_pattern):
    row_encoding = row[encoding_cols_combined].fillna('').infer_objects(copy=False)
    for _, pattern_row in pattern_data.iterrows():
        pattern_encoding = pattern_row[encoding_cols_pattern].fillna('').infer_objects(copy=False)
        if np.array_equal(row_encoding.values, pattern_encoding.values):
            element_class = pattern_row.get('element_class', None)
            if not element_class or pd.isna(element_class):
                element_class = pattern_row.get('fragment_name', 'Unknown')
            return element_class, 1
    return None, 0

def match_patterns(df, patterns):
    if isinstance(patterns, list):
        patterns = pd.DataFrame(patterns)
    presence_encoding_cols_combined = [col for col in df.columns if col.startswith("Level_")]
    presence_encoding_cols_pattern = [col for col in patterns.columns if col.startswith("Level_")]
    df[['element_class', 'match']] = df.apply(lambda row: pd.Series(compare_fragment(row, patterns, presence_encoding_cols_combined, presence_encoding_cols_pattern)), axis=1)
    return df

def extract_zip_and_combine_chapters(zip_path):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(TEMP_FOLDER)
        combined_html = ""
        for root, dirs, files in os.walk(TEMP_FOLDER):
            if os.path.basename(root).lower().startswith('chapter'):
                for file in files:
                    if file.endswith('.html') or file.endswith('.xhtml'):
                        html_path = os.path.join(root, file)
                        with open(html_path, 'r', encoding='utf-8') as f:
                            combined_html += f.read()
        return combined_html
    except Exception as e:
        print(f"Error extracting and processing ZIP file: {e}")
        return None

import re
from collections import defaultdict

def extract_specific_tag_classes(fragment, current_tag):
    tag_specific_classes = []
     
    # attr_segment = fragment.attributes.strip()
    for attr_segment in fragment.attributes.split(';'):
        
        if 'class' in attr_segment and current_tag in attr_segment:
            
            matches = re.findall(r"\'([^\']+)\'", attr_segment)

            
            tag_specific_classes.extend(matches)

    return sorted(set(tag_specific_classes))


def structure_clusters(df):
    clusters = {}
    noise = []

   
    element_types = set(fragment.element_type for fragment in df.itertuples(index=False))

    for element_type in element_types:
        cluster_num = next((num for num in range(1, 100) if f"cluster{num}" not in clusters), None)

        if cluster_num is None:
            cluster_num = max(int(k.split('cluster')[-1]) for k in clusters.keys()) + 1

        cluster_data = [fragment for fragment in df.itertuples(index=False) if fragment.element_type == element_type]

        if not cluster_data:
            print(f"No data for element type {element_type}")
        else:
            cluster_groups = defaultdict(list)

            for fragment in cluster_data:
                element_class = fragment.element_class

                level_encodings = {}
                for attr in dir(fragment):
                    if attr.startswith("Level_") and getattr(fragment, attr) == 1:
                        level_tag = attr.split("_")[2]
                        print(level_tag)
                        classes = extract_specific_tag_classes(fragment, level_tag)
                       
                        level_encodings[attr] = {
                            "value": 1,
                            "class": classes
                        }

                if isinstance(element_class, float) and np.isnan(element_class):
                    noise_fragment = {
                        "fragment_id": fragment.fragment_id,
                        "fragment_name": fragment.fragment_name,
                        "Level": fragment.Level,
                        "element_type": fragment.element_type,
                        "element_class": "",
                        "full_attributes": fragment.attributes,
                        **level_encodings
                    }
                    noise.append(noise_fragment)
                    continue

                filtered_fragment = {
                    "fragment_id": fragment.fragment_id,
                    "fragment_name": fragment.fragment_name,
                    "Level": fragment.Level,
                    "element_type": fragment.element_type,
                    "element_class": element_class or "",
                    "full_attributes": fragment.attributes,
                    **level_encodings
                }

                group_key = f"{element_class}" if element_class else "unknown"

                cluster_groups[group_key].append(filtered_fragment)

            if cluster_groups:
                clusters[f"cluster{cluster_num}"] = dict(cluster_groups)

    return clusters, noise


def collect_fragments(fragments_df):
    fragments = []
    for _, row in fragments_df.iterrows():
        filtered_encoding = {}
        for k, v in row.items():
            if k.startswith("Level_") and v == 1:
                level = int(k.split('_')[1])
                tag = k.split('_')[2]
                filtered_encoding[k] = {'value': v}
        
        fragment_details = {
            'fragment_id': row['fragment_id'],
            'fragment_name': row['fragment_name'],
            'Level': row['Level'],
            'element_type': row['element_type'],
            'element_class': row['element_class'],
            'attributes': row['attributes'],
            **filtered_encoding
        }
        fragments.append(fragment_details)
    
    return fragments


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploadtemp', methods=['POST'])
def upload_files1():
    try:
        if 'zip_file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        zip_file = request.files['zip_file']
        if zip_file and zip_file.filename.endswith('.zip') or zip_file.filename.endswith('.epub'):
            filename = secure_filename(zip_file.filename)
            zip_path = os.path.join(UPLOAD_FOLDER, filename)
            zip_file.save(zip_path)
            start_time = time.time()
            combined_html = extract_zip_and_combine_chapters(zip_path)
            if not combined_html:
                return jsonify({'error': 'Failed to extract and combine chapters from ZIP file'}), 500
            
            initial_fragment_id = 0
            fragment_name = os.path.splitext(filename)[0]
            soup = BeautifulSoup(combined_html, 'html.parser')
            initial_parent_tag = soup.find('section').name if soup.find('section') else 'body'
            
            fragment_data, initial_fragment_id = extract_fragments(combined_html, initial_fragment_id, fragment_name, initial_parent_tag)
            if not fragment_data:
                return jsonify({'error': 'No valid HTML files processed'}), 400
            columns = ["fragment_id", "fragment_name", "Level", "element_type", "element_class", "attributes"] + [f'Level_{lvl}_{tag}' for lvl in levels for tag in tags]
            df_combined = pd.DataFrame(fragment_data, columns=columns)
            with open(PATTERN_FILE_PATH, 'r') as f:
                pattern_data = json.load(f)
            df_combined = match_patterns(df_combined, pattern_data)
            df_clustered = perform_clustering(df_combined)
            clusters, noise = structure_clusters(df_clustered)
            output_data = {'clusters': clusters, 'noise': noise}
            output_filename = os.path.join(CLUSTER_OUTPUT_FOLDER, f"clustered_{filename}.json")
            with open(output_filename, 'w') as f:
                json.dump(output_data, f, indent=4)
            end_time = time.time()
            logging.info(f"Clustering completed in {end_time - start_time:.2f} seconds")
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Invalid file format. Please upload a ZIP file or EPUB file.'}), 400
    except Exception as e:
        logging.error(f"Error occurred during clustering: {e}")
        return jsonify({'error': 'An error occurred during clustering'}), 500

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        if 'zip_file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        zip_file = request.files['zip_file']
        if zip_file and zip_file.filename.endswith('.zip') or zip_file.filename.endswith('.epub'):
            filename = secure_filename(zip_file.filename)
            zip_path = os.path.join(UPLOAD_FOLDER, filename)
            zip_file.save(zip_path)
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Invalid file format. Please upload a ZIP file.'}), 400
            # start_time = time.time()
            # combined_html = extract_zip_and_combine_chapters(zip_path)
            # if not combined_html:
            #     return jsonify({'error': 'Failed to extract and combine chapters from ZIP file'}), 500
            
            # initial_fragment_id = 0
            # fragment_name = os.path.splitext(filename)[0]
            # soup = BeautifulSoup(combined_html, 'html.parser')
            # initial_parent_tag = soup.find('section').name if soup.find('section') else 'body'
            
            # fragment_data, initial_fragment_id = extract_fragments(combined_html, initial_fragment_id, fragment_name, initial_parent_tag)
            # if not fragment_data:
            #     return jsonify({'error': 'No valid HTML files processed'}), 400
            # columns = ["fragment_id", "fragment_name", "Level", "element_type", "element_class", "attributes"] + [f'Level_{lvl}_{tag}' for lvl in levels for tag in tags]
            # df_combined = pd.DataFrame(fragment_data, columns=columns)
            # with open(PATTERN_FILE_PATH, 'r') as f:
            #     pattern_data = json.load(f)
            # df_combined = match_patterns(df_combined, pattern_data)
            # df_clustered = perform_clustering(df_combined)
            # clusters, noise = structure_clusters(df_clustered)
            # output_data = {'clusters': clusters, 'noise': noise}
            # output_filename = os.path.join(CLUSTER_OUTPUT_FOLDER, f"clustered_{filename}.json")
            # with open(output_filename, 'w') as f:
            #     json.dump(output_data, f, indent=4)
            # end_time = time.time()
            # logging.info(f"Clustering completed in {end_time - start_time:.2f} seconds")

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/extract", methods=["GET"])
def extract():
    try:
        zip_path = os.path.join(UPLOAD_FOLDER, os.listdir(UPLOAD_FOLDER)[0])
        html = extract_zip_and_combine_chapters(zip_path)
        extract = os.path.join(TEMP_FOLDER, os.listdir(TEMP_FOLDER)[0])
        if html:
            with open(extract+".html", 'w+', encoding="utf-8") as f:
                f.write(html)
            shutil.rmtree(extract)
            return jsonify({"status": "success"}), 200
        else:
            return jsonify({"status": "success", "error":"html data not found"}), 200
    except Exception as e:
        return jsonify({"status": "filed", "error": "internal server error"})

@app.route('/epubfragment', methods=['GET'])
def fragment():
    start_time = time.time()
    initial_fragment_id = 0
    fragment_name = os.listdir(TEMP_FOLDER)[0]
    with open(os.path.join(TEMP_FOLDER,fragment_name), 'r', encoding="utf-8") as f:
        combined_html = f.read()
    soup = BeautifulSoup(combined_html, 'html.parser')
    initial_parent_tag = soup.find('section').name if soup.find('section') else 'body'
    
    fragment_data, initial_fragment_id = extract_fragments(combined_html, initial_fragment_id, fragment_name, initial_parent_tag)
    if not fragment_data:
        return jsonify({'error': 'No valid HTML files processed'}), 400
    columns = ["fragment_id", "fragment_name", "Level", "element_type", "element_class", "attributes"] + [f'Level_{lvl}_{tag}' for lvl in levels for tag in tags]
    df_combined = pd.DataFrame(fragment_data, columns=columns)
    with open(PATTERN_FILE_PATH, 'r') as f:
        pattern_data = json.load(f)
    df_combined = match_patterns(df_combined, pattern_data)
    df_clustered = perform_clustering(df_combined)
    clusters, noise = structure_clusters(df_clustered)
    output_data = {'clusters': clusters, 'noise': noise}
    output_filename = os.path.join(CLUSTER_OUTPUT_FOLDER, f"clustered_{fragment_name}.json")
    with open(output_filename, 'w') as f:
        json.dump(output_data, f, indent=4)
    end_time = time.time()
    logging.info(f"Clustering completed in {end_time - start_time:.2f} seconds")
    return jsonify({'status':'success'})

if __name__ == '__main__':
    app.run(debug=False)
