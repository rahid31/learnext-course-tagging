import json
import torch
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from ast import literal_eval
import pandas as pd

import streamlit as st

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

class TagClassifier:
    def __init__(self, tag_file="tag_guidelines.json", openai_model="gpt-4"):
        # Load tag guidelines from a single JSON file
        with open(tag_file, "r", encoding="utf-8") as f:
            tag_guidelines = json.load(f)

        # Flatten into (tag, full_text) list
        self.flat_tag_guidelines = [(tag, f"{tag}: {desc}") for tag, desc in tag_guidelines.items()]

        self.tag_names = [item[0] for item in self.flat_tag_guidelines]
        self.tag_texts = [item[1] for item in self.flat_tag_guidelines]

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.tag_embeddings = self.embedder.encode(self.tag_texts, convert_to_tensor=True)

        self.client = OpenAI()
        self.model = openai_model

    def generate_prompt(self, course_name, description, tag_guidelines):
        tag_text = "\n".join(f"- {tag}: {desc}" for tag, desc in tag_guidelines.items())
        return f"""
You are an assistant that classifies training courses into appropriate categories based on course name and description.

### Step-by-step Instructions:
1. Read the course name and description.
2. Think about which tags best match the course content. Try to translate into one language before applying.
3. From the list, choose only tags that clearly apply (ignore weak matches).
4. DO NOT explain your answer. Only output the Python list of applicable tags (e.g., ['Leadership', 'Problem Solving']). If none apply, return [].

### Course Name:
{course_name}

### Description:
{description}

### Tag Guidelines:
{tag_text}
""".strip()

    def query_openai(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that categorizes educational content based on predefined tags."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    def classify(self, course_name: str, description: str):
        # Embed course text
        course_embedding = self.embedder.encode(f"{course_name}. {description}", convert_to_tensor=True)

        # Similarity scores & pick top N tags
        scores = util.pytorch_cos_sim(course_embedding, self.tag_embeddings)[0]
        top_n = 7
        top_indices = torch.topk(scores, top_n).indices.cpu().tolist()

        filtered_tags = {
            self.tag_names[i]: self.tag_texts[i].split(": ", 1)[1]
            for i in top_indices
        }

        prompt = self.generate_prompt(course_name, description, filtered_tags)

        try:
            response_text = self.query_openai(prompt)
            ai_tags = literal_eval(response_text)
        except Exception as e:
            ai_tags = []
            print(f"Error during classification: {e}")

        return ai_tags
    
class OtherClassifier:
    def __init__(self, tag_file="other_guidelines.json", openai_model="gpt-4"):
        # Load tag guidelines from a single JSON file
        with open(tag_file, "r", encoding="utf-8") as f:
            tag_guidelines = json.load(f)

        # Flatten into (tag, full_text) list
        self.flat_tag_guidelines = [(tag, f"{tag}: {desc}") for tag, desc in tag_guidelines.items()]

        self.tag_names = [item[0] for item in self.flat_tag_guidelines]
        self.tag_texts = [item[1] for item in self.flat_tag_guidelines]

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.tag_embeddings = self.embedder.encode(self.tag_texts, convert_to_tensor=True)

        self.client = OpenAI()
        self.model = openai_model

    def generate_prompt(self, course_name, description, tag_guidelines):
        tag_text = "\n".join(f"- {tag}: {desc}" for tag, desc in tag_guidelines.items())
        return f"""
You are an assistant that classifies training courses into appropriate categories based on course name and description.

### Step-by-step Instructions:
1. Read the course name and description.
2. Think about which tags best match the course content. Try to translate into one language before applying.
3. From the list, choose only tags that clearly apply (ignore weak matches).
4. DO NOT explain your answer. Only output the Python list of applicable tags (e.g., ['Leadership', 'Problem Solving']). If none apply, return [].

### Course Name:
{course_name}

### Description:
{description}

### Tag Guidelines:
{tag_text}
""".strip()

    def query_openai(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that categorizes educational content based on predefined tags."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    def classify(self, course_name: str, description: str):
        # Embed course text
        course_embedding = self.embedder.encode(f"{course_name}. {description}", convert_to_tensor=True)

        # Similarity scores & pick top N tags
        scores = util.pytorch_cos_sim(course_embedding, self.tag_embeddings)[0]
        top_n = 7
        top_indices = torch.topk(scores, top_n).indices.cpu().tolist()

        filtered_tags = {
            self.tag_names[i]: self.tag_texts[i].split(": ", 1)[1]
            for i in top_indices
        }

        prompt = self.generate_prompt(course_name, description, filtered_tags)

        try:
            response_text = self.query_openai(prompt)
            ai_tags = literal_eval(response_text)
        except Exception as e:
            ai_tags = []
            print(f"Error during classification: {e}")

        return ai_tags
    
class BulkTagging:
    def __init__(self, tag_file="tag_guidelines.json", openai_model="gpt-4"):
            # Load tag guidelines from a single JSON file
            with open(tag_file, "r", encoding="utf-8") as f:
                tag_guidelines = json.load(f)

            # Flatten into (tag, full_text) list
            self.flat_tag_guidelines = [(tag, f"{tag}: {desc}") for tag, desc in tag_guidelines.items()]

            self.tag_names = [item[0] for item in self.flat_tag_guidelines]
            self.tag_texts = [item[1] for item in self.flat_tag_guidelines]

            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.tag_embeddings = self.embedder.encode(self.tag_texts, convert_to_tensor=True)

            self.client = OpenAI()
            self.model = openai_model

    def generate_prompt(self, course_name, description, tag_guidelines):
        tag_text = "\n".join(f"- {tag}: {desc}" for tag, desc in tag_guidelines.items())
        return f"""
You are an assistant that classifies training courses into appropriate categories based on course name and description.

### Step-by-step Instructions:
1. Read the course name and description.
2. Think about which tags best match the course content. Try to translate into one language before applying.
3. From the list, choose only tags that clearly apply (ignore weak matches).
4. DO NOT explain your answer. Only output the Python list of applicable tags (e.g., ['Leadership', 'Problem Solving']). If none apply, return [].

### Course Name:
{course_name}

### Description:
{description}

### Tag Guidelines:
{tag_text}
""".strip()

    def query_openai(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that categorizes educational content based on predefined tags."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    def classify_bulk(self, df, name_col="name", desc_col="description", output_path=None):
        results = []
        for idx, row in df.iterrows():

            course_name = row[name_col]
            description = row[desc_col]
            print(f"[{idx+1}/{len(df)}] Tagging: {course_name}")

            course_embedding = self.embedder.encode(f"{course_name}. {description}", convert_to_tensor=True)
            scores = util.pytorch_cos_sim(course_embedding, self.tag_embeddings)[0]
            top_indices = torch.topk(scores, 7).indices

            filtered_tags = {
                self.tag_names[i]: self.tag_texts[i].split(": ", 1)[1]
                for i in top_indices
            }

            prompt = self.generate_prompt(course_name, description, filtered_tags)

            try:
                response_text = self.query_openai(prompt)
                ai_tags = literal_eval(response_text)
            except Exception as e:
                print(f"Error tagging course: {e}")
                ai_tags = []

            all_tags = [t[0] for t in self.flat_tag_guidelines]
            ai_row = {'course': course_name}
            ai_row.update({tag: tag in ai_tags for tag in all_tags})
            results.append(ai_row)

        result_df = pd.DataFrame(results)

        return result_df
    
