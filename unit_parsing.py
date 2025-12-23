import pandas as pd
import json
import re
import os
import time
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Define file paths
input_file = '####'
output_file = '####'

# Google Gemini API Key
genai.configure(api_key="####")
model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")
generation_config = genai.GenerationConfig(temperature=0.0, max_output_tokens=8192)

def clean_and_parse_json(response_text):
    """
    Cleans and parses the JSON response text from the API.
    """
    try:
        # Remove extra characters or non-JSON prefixes
        cleaned_text = response_text.strip(' \n')
        if cleaned_text.lower().startswith('json'):
            cleaned_text = cleaned_text[4:].strip()

        # Match and extract JSON content
        json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            raise ValueError("No valid JSON object found.")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSONDecodeError: {e}")


def generate_prompt(group, group_index):
    """
    Generates a GPT prompt for a given group of reports.
    """
    combined_reports = "\n---REPORT SEPARATOR---\n".join(group)
    prompt = f"""
    This word string is a CT reports that have undergone de-identification and preprocessing. 

    Your task:
    1. Correct typos in the word strings to their most likely intended forms based on medical terminology.
       - Ensure that anatomical expressions use the correct parts of speech. For example, "mediastinum lymph node" should be corrected to "mediastinal lymph node."
       - If anatomical locations are connected by "and" or "or", explicitly expand them to ensure each location is fully described:
         - Example: "right internal mammary and left axillary lymph node" should be expanded to "right internal mammary lymph node and left axillary lymph node."
    2. Divide the corrected word strings into **concise lexicon units** and assign each unit to one of the following categories:
       - **1. Anatomical entity with location**: Anatomical structures combined with their positional descriptions (e.g., "upper lobe of right lung", "superior pole of left kidney", "mediastinal lymph node").
           - **Do not split anatomical components** such as "lung, lobe, segment" or similar hierarchical descriptions into separate units. These must be combined into a single lexicon unit.
       - **2. Physiologic condition**: Functional or pathological states or processes occurring within the body. These are inherent conditions (e.g., "hyperinflation", "consolidation", "fibrosis", "granuloma", "cyst", "bronchiectasis", "atelectasis", "lymphadenopathy", "coronary artery calcification") or **symptoms** such as "cough", "pain", or "shortness of breath" when directly stated in the text. These are **not explicitly described as visual observations** on imaging.
       - **3. Imaging observation**: Findings or abnormalities described as direct **visual interpretations** from imaging (e.g., "ill-defined margin", "nodular opacity", "ground-glass pattern"). These are descriptive terms that indicate how a condition appears in imaging studies.
           - **Key distinction**:
             - If the term refers to a condition inherently existing in the body (e.g., "fibrosis", "consolidation", "lymphadenopathy"), it belongs to **Physiologic condition**.
             - If the term refers to how the condition is visually described on imaging (e.g., "ground-glass opacity", "nodular appearance"), it belongs to **Imaging observation**.
           - Example:
             - "Fibrosis" → **Physiologic condition**
             - "Reticular pattern of fibrosis" → **Imaging observation**
             - "Nodular opacity" → **Imaging observation**
             - "Pulmonary nodules" → **Imaging observation**
             - "Chronic interstitial pneumonia" → **Physiologic condition**
       - **4. Physical object**: Any external or internal object mentioned in the report (e.g., "stent", "catheter", "surgical clip").
           - **Important clarification**: Physical object must refer to an artificially introduced or external structure. Natural formations within the body, even if they resemble objects (e.g., stones, calculi), should not be categorized here. Instead, classify them as 2. Physiologic condition if they indicate a pathological state.
       - **5. Procedure**: Any medical or surgical process or action (e.g., "biopsy", "contrast-enhanced CT scan", "follow up procedure").
       - **6. Others**: Use this category if the unit does not fit into the above categories (e.g., "clinical information section") or the meaning is unclear.

    3. Follow these **Important Rules** when creating the lexicon units:
       - A single lexicon unit **must not mix categories**. For example:
         - Incorrect: "renal mass and biopsy procedure".
         - Correct: ["renal mass", "biopsy procedure"].
       - Findings and locations must be **split into separate units**:
         - Example 1: "consolidation in lower lobe of right lung" → ["consolidation", "lower lobe of right lung"].
         - Example 2: "nodular opacity in upper lobe of left lung" → ["nodular opacity", "upper lobe of left lung"].
       - **Handle conjunctions properly**:
         - If items are connected by "and", "or", or similar conjunctions, split them into separate units:
           - Example: "biopsy or surgery" → ["biopsy", "surgery"].
         - If conjunctions are missing but implied, infer the separation:
           - Example: "diffuse ground-glass opacity consolidation nodular opacity" → ["diffuse ground-glass opacity", "consolidation", "nodular opacity"].
         - For anatomical locations connected by "and" or "or", ensure each is expanded to a fully described location before splitting:
           - Example: "right internal mammary and left axillary lymph node" → ["right internal mammary lymph node", "left axillary lymph node"]. 
           
       - **Avoid overly long units**:
         - Long expressions should be split into smaller meaningful components:
           - Example: "low attenuating lesion in right thyroid gland" → ["low attenuating lesion", "right thyroid gland"].

    4. Always ensure:
       - The original word order is preserved.
       - Typos are corrected, and meaningless words are removed or replaced during the correction process.
       - findings and locations must be split into separate units

    ### Additional Guidance for Ambiguous Cases:
    - When terms seem ambiguous, follow these guidelines:
      1. **Check for explicit imaging-related descriptors**:
         - Words like "opacity", "pattern", "margin", "enhancement" often indicate **Imaging observation**.
      2. **Default to Physiologic condition**:
         - If a term could describe a general condition without clear imaging context, assign it to **Physiologic condition**.
      3. **Complex units**: 
         - Break down terms with both a visual and physiologic aspect:
           - Example: "reticular opacity of lung fibrosis" → ["reticular opacity", "lung fibrosis"].

Format the output as JSON:
{{
  "reports": [
    {{
      "report_index": <index within group>,
      "lexicon_units": [
        {{"unit": "<unit1>", "category": <category_number>}},
        {{"unit": "<unit2>", "category": <category_number>}},
        ...
      ]
    }}
    ...
  ]
}}

Example reports:
Report 1: "reticular opacity and consolidation in lower lobe of right lung superior segment bronchial wall thickening and centrilobular nodule in upper lobe of left lung peripheral portion"
Expected output:
{{
  "report_index": 1,
  "lexicon_units": [
    {{"unit": "reticular opacity", "category": 3}},
    {{"unit": "consolidation", "category": 2}},
    {{"unit": "lower lobe of right lung superior segment", "category": 1}},
    {{"unit": "bronchial wall thickening", "category": 3}},
    {{"unit": "centrilobular nodule", "category": 3}},
    {{"unit": "upper lobe of left lung", "category": 1}},
    {{"unit": "peripheral portion", "category": 1}},
  ]
}}

Reports:
{combined_reports}
"""
    return prompt.strip()


def process_group(group, group_index):
    """
    Processes a single group of reports by sending it to the GPT API and parsing the response.
    """
    max_retries = 10
    retry_delay = 2
    prompt = generate_prompt(group, group_index)
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
            if "reports" not in response_data:
                raise ValueError("Invalid response format: 'reports' key not found.")
            return response_data["reports"]
        except Exception as e:
            print(f"Error processing group {group_index}, attempt {attempt + 1}: {e}")

            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to process group {group_index} after {max_retries} attempts.")
                return []

def append_to_excel(group_index, group_results, output_df):
    """
    Appends processed group results to an Excel file.
    """



def process_reports(input_file, output_file, reports_per_group):
    reports_df = pd.read_excel(input_file, header=None, names=['Expanded_Report'])

    # Group reports
    grouped_reports = [
        reports_df['Expanded_Report'][i:i+reports_per_group].tolist()
        for i in range(0, len(reports_df), reports_per_group)
    ]

    # Check if the file already exists
    if os.path.exists(output_file):
        output_df = pd.read_excel(output_file)
    else:
        output_df = pd.DataFrame(columns=['Group Index', 'Report Index', 'Unit', 'Category'])

    all_results = []  # Store all results
    start_time = time.time()  # Start the timer for cumulative processing

    for group_index, group in enumerate(grouped_reports):
        group_start_time = time.time()  # Start time for this group
        results = process_group(group, group_index)
        group_end_time = time.time()  # End time for this group

        for report in results:
            for unit in report["lexicon_units"]:
                all_results.append({
                    'Group Index': group_index,
                    'Report Index': report["report_index"],
                    'Unit': unit["unit"],
                    'Category': unit["category"]
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

process_reports(input_file, output_file, reports_per_group=5)
