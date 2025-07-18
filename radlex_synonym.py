import openai
import pandas as pd
import numpy as np

import os
import time
import json
import re

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Google Gemini API Key
genai.configure(api_key="####")
model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")
generation_config = genai.GenerationConfig(temperature=0.0, max_output_tokens=8192)

def generate_prompt(group):
    """
    Generates a Gemini prompt for a given group of reports.
    """
    combined_terms = "\n---TERM SEPARATOR---\n".join(group)
    prompt = f"""
    RadLex is a comprehensive set of radiology terms for use in radiology reporting, decision support, data mining, data registries, education, and research. It is widely used in medical imaging, artificial intelligence, and clinical decision support systems to ensure consistent and precise descriptions of radiological findings.

    However, the current synonym structure in RadLex is relatively **rigid and limited**, which restricts its applicability in diverse real-world clinical and AI-driven scenarios. Expanding and refining synonym mappings is essential to enhance its usability in **natural language processing (NLP), deep learning models, and automated clinical decision support systems**.

    Generate synonyms and lexical variants for the following RadLex lexicon terms and categorize them into **four distinct groups**. 
    **Important: Each generated synonym or lexical variant must fully capture the complete meaning of the original term as a complete phrase. Do not extract or generate only a partial component of the term.**  
    **Before finalizing your response, double-check that every generated synonym or lexical variant fully encapsulates the complete clinical concept of the original term. If any of the outputs do not meet this requirement, please revise them accordingly.**  
    **Return only the JSON object without any extra text or commentary.**
    The expressions must be clinically relevant, medically precise, and commonly used in medical literature or practice.
    
    ### **Definition: Synonyms & Lexical Variants**
    For the purpose of this task, **"synonyms"** refer strictly to terms that are **semantically equivalent and can be used interchangeably in all clinical contexts.**  
    **"Lexical variants"** include morphological, orthographic, and abbreviation variations, which differ in form but not in meaning.
    
    ### **Categories of Synonyms & Lexical Variants:**
    1. **Morphological Variants (Category 1):**  
       - Terms that are **fully synonymous but differ in grammatical form** (e.g., noun vs. adjective, singular vs. plural, verb vs. participle).
       - **Examples:**
         - pleura vs. pleural  
         - bronchiectasis vs. bronchiectatic  
         - attenuation vs. attenuated vs. attenuating  
    
    2. **Orthographic Variants (Category 2):**  
       - Terms that are **fully synonymous but differ only in spacing, hyphenation, or alternative spellings**.
       - **Examples:**
         - air trapping vs. air-trapping vs. airtrapping  
         - airspace vs. air space vs. air-space  
    
    3. **Acronyms & Abbreviations (Category 3):**  
       - Commonly used abbreviations or acronyms that are synonymous with the term.  
       - **Examples:**
         - myocardial infarction → MI  
         - acute respiratory distress syndrome → ARDS  
    
    4. **Strict Semantic Synonyms (Category 4):**  
       - Terms that **convey the exact same meaning and can be used interchangeably in all clinical contexts**.
       - **Synonyms must be strictly equivalent and should not introduce ambiguity or potential contextual differences.**
       - **Examples:**
         - shortness of breath vs. dyspnea  
         - neoplasm vs. tumorous condition  
         - probably vs. likely  
    
Format the output as JSON:
{{
  "term_and_synonyms": [
    {{
      "term": "<lexicon 1>",
      "category_1": ["Morphological Variant 1", "Morphological Variant 2", "Morphological Variant 3", ...],
      "category_2": ["Orthographic Variant 1", "Orthographic Variant 2", "Orthographic Variant 3", ...],
      "category_3": ["Acronym 1", "Acronym 2","Acronym 3", ...],
      "category_4": ["Strict Semantic Synonym 1", "Strict Semantic Synonym 2", "Strict Semantic Synonym 3", ...]
    }}
    ...
  ]
}}

terms:
{combined_terms}
"""
    return prompt.strip()

def process_group(group, group_index):
    """
    Processes a single group of lexicon by sending it to the gemini API and parsing the response.
    """
    prompt = generate_prompt(group)
    max_retries = 10
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            # Send the API request

            response = model.generate_content([prompt], generation_config=generation_config,
                                              safety_settings={
                                                  HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                                  HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                                  HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                                  HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
                                              }
                                              )

            # Validate if the response is empty
            if not response.text:
                raise ValueError("Received an empty response from the API.")

            # Parse the JSON response content
            response_json = response.text

            # Remove comments or non-JSON parts using regular expressions
            # This will match and extract the JSON part within the response
            json_match = re.search(r"\{.*\}", response_json, re.DOTALL)

            if json_match:
                # Extract the JSON part
                response_json = json_match.group(0)
                response_data = json.loads(response_json)  # Parsing the string as JSON
            else:
                raise ValueError("No valid JSON object found in the response.")

            # Validate response structure
            if "term_and_synonyms" not in response_data:
                raise ValueError("Invalid response format: 'term_and_synonyms' key not found.")
            return response_data["term_and_synonyms"]

        except Exception as e:
            print(f"Error processing group {group_index}, attempt {attempt + 1}: {e}")

            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to process group {group_index} after {max_retries} attempts.")
                return [{"term": "error", "category_1": "error", "category_2": "error",
                         "category_3": "error", "category_4": "error"}]

radlex_file = '###'
output_file = '###'

def clean_synonyms(synonyms_list):
    """Converts a list of synonyms into a clean string separated by '|' without brackets or quotes."""
    if isinstance(synonyms_list, list):
        return "|".join(synonyms_list)
    return ""  # Return an empty string if no synonyms are present


def process_lexicons(input_file, output_file, lexicon_per_group):
    reports_df = pd.read_excel(input_file, sheet_name='Sheet1', dtype=str, keep_default_na=False)
    reports_df['Preferred Label'] = reports_df['Preferred Label'].astype(str)

    # Group reports
    grouped_lexicons = [
        reports_df['Preferred Label'][i:i + lexicon_per_group].tolist()
        for i in range(0, len(reports_df), lexicon_per_group)
    ]

    # Check if the file already exists
    if os.path.exists(output_file):
        output_df = pd.read_excel(output_file)
    else:
        output_df = pd.DataFrame(columns=['term', 'category_1', 'category_2', 'category_3', 'category_4'])

    all_results = []  # Store all results
    start_time = time.time()  # Start the timer for cumulative processing

    for group_index, group in enumerate(grouped_lexicons):
        group_start_time = time.time()  # Start time for this group
        results = process_group(group, group_index)
        group_end_time = time.time()  # End time for this group

        for result in results:
            all_results.append({
                "term": result["term"],
                "category_1": clean_synonyms(result["category_1"]),
                "category_2": clean_synonyms(result["category_2"]),
                "category_3": clean_synonyms(result["category_3"]),
                "category_4": clean_synonyms(result["category_4"]),
            })
        # Calculate and print timing information
        elapsed_group_time = group_end_time - group_start_time
        elapsed_total_time = group_end_time - start_time
        print(f"Finished processing group {group_index}. "
              f"Group time: {elapsed_group_time:.2f} seconds, "
              f"Total elapsed time: {elapsed_total_time:.2f} seconds.")

        # Export to Excel every 50 groups
        if (group_index + 1) % 20 == 0:
            print(f"Exporting results for groups up to {group_index}...")
            temp_df = pd.DataFrame(all_results)
            if os.path.exists(output_file):
                existing_df = pd.read_excel(output_file)
                temp_df = pd.concat([existing_df, temp_df], ignore_index=True)
            temp_df.to_excel(output_file, index=False)
            all_results = []  # Clear intermediate results after export

    # Export any remaining results
    if all_results:
        print("Exporting remaining results...")
        temp_df = pd.DataFrame(all_results)
        if os.path.exists(output_file):
            existing_df = pd.read_excel(output_file)
            temp_df = pd.concat([existing_df, temp_df], ignore_index=True)
        temp_df.to_excel(output_file, index=False)

    total_elapsed_time = time.time() - start_time
    print(f"Processing completed. Total elapsed time: {total_elapsed_time:.2f} seconds.")

# Process the reports
process_lexicons(radlex_file, output_file, lexicon_per_group=15)