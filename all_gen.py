import google.generativeai as genai # needs new protobuf (update if needed)
import time
import pandas as pd
from openai import OpenAI
import requests
from google.generativeai.types import HarmCategory, HarmBlockThreshold


def preprocess_formula(formula):
    formula = formula.replace("~", "¬")
    formula = formula.replace("&", "∧")
    formula = formula.replace("|", "∨")
    formula = formula.replace("$", "→")
    formula = formula.replace("@", "∀")
    formula = formula.replace("/", "∃")
    return formula

exp = """
SameSize ( x , y ): x and y are the same size.
Smaller ( x , y ): x is smaller than y.
SameCol ( x , y ): x and y are in the same column.
Larger ( x , y ): x is larger than y.
BackOf ( x , y ): x is behind y.
Medium ( x ): x is medium.
Large ( x ): x is large.
FrontOf ( x , y ): x is in front of y.
Adjoins ( x , y ): x adjoins y.
Small ( x ): x is small.
Between ( x , y , z ): x is between y and z.
LeftOf ( x , y ): x is to the left of y.
Cube ( x ): x is a cube.
Dodec ( x ): x is a dodecahedron.
RightOf ( x , y ): x is to the right of y.
SameRow ( x , y ): x and y are in the same row.
SameShape ( x , y ): x and y are the same shape.
Tet ( x ): x is a tetrahedron.
"""

#llama
API_TOKEN = ""

def model_inference(model, datapoint, token=API_TOKEN):
    model = model
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}

    parameters = {"max_new_tokens": 250, "temperature": 0.1, "top_k" : 1}  # "return_full_text": False, 
    options = {"wait_for_model": True}  # set to True if you want to wait for the model to be loaded

    def query_model(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    input_data = datapoint
    prompt = f"Translate the following formula into English.\nThe following is the meaning of the predicates used in the formula:\n{exp}\nONLY RETURN THE TRANSLATION. DO NOT USE LOGICAL SYMBOLS. DO NOT GIVE ANY EXPLANATION.\n\nFormula: {input_data}\nTranslation:"

    output = query_model(
        {
            "inputs": prompt,
            "parameters": parameters,
            "options": options
        }
    )


    try:
        translation = output[0]["generated_text"]
        translation = translation.strip().replace("\n", " ")
    except:
        translation = output

    print(translation)
    return translation


#GPT
client = OpenAI(api_key="")

def gpt_inference(input_data):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Translate the following formula into English.\nThe following is the meaning of the predicates used in the formula:\n{exp}\nONLY RETURN THE TRANSLATION. DO NOT USE LOGICAL SYMBOLS. DO NOT GIVE ANY EXPLANATION.\n\nFormula: {input_data}\nTranslation:",
            }
        ],
        model="gpt-4o-mini",
        temperature=0,
    )
    print(chat_completion)
    return chat_completion.choices[0].message.content


#gemini
genai.configure(api_key="AIzaSyCtAwWrdUsOauw3Ar6RAEPskju0zwGQekE")
gemini = genai.GenerativeModel("gemini-1.5-flash")

def gemini_inference(input_data):
    prompt = f"Translate the following formula into English.\nThe following is the meaning of the predicates used in the formula:\n{exp}\nONLY RETURN THE TRANSLATION. DO NOT USE LOGICAL SYMBOLS. DO NOT GIVE ANY EXPLANATION.\n\nFormula: {input_data}\nTranslation:"
    try:
        response = gemini.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0),safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,}) # deterministic
        if response.text:
            print(response.text)
            return response.text
        else:
            print(f"Response is invalid. Finish reason: {response.finish_reason}")
            return "ERROR"
    except Exception as e:
        print(f"Error occurred: {e}")
        return "ERROR"
    
    


#produce cvs file of outputs
df = pd.read_csv("dataset_sub.csv")
data = {
    "nested": df['nested'].values.tolist(),
    "connectives": df['connectives'].values.tolist(),
    "negation": df['negation'].values.tolist()
}

final_data = []

for subset, lst in data.items():
    for formula in lst:
        input = formula
        
        output = str(gemini_inference(formula))
        final_data.append((formula, output, "gemini-1.5-flash",subset))

        output = model_inference("codellama/CodeLlama-34b-Instruct-hf", input, token=API_TOKEN)
        final_data.append((input, output, "codellama/CodeLlama-34b-Instruct-hf",subset))
        
        output = gpt_inference(formula)
        final_data.append((formula, output, "gpt-4o-mini",subset))


df_data = pd.DataFrame(final_data, columns=["input", "output", "model"])
# clean codellama outputs
df_data["output"] = df_data["output"].apply(lambda x: [x.split("Answer: ")[1] if "Answer" in x and isinstance(x, str) else x][0])
#strip any whitespace at beginning or end
df_data["output"] = df_data["output"].apply(lambda x: [x.strip() if isinstance(x, str) else x][0])

df_data.to_csv("ill_outputs.csv", index=False)



df = pd.read_csv("dataset.csv")
for formula in df['0']:
    output = str(gemini_inference(formula))
    final_data.append((formula, output, "gemini-1.5-flash"))
for formula in df['0']:
    output = model_inference("codellama/CodeLlama-34b-Instruct-hf", formula, token=API_TOKEN)
    final_data.append((formula, output, "codellama/CodeLlama-34b-Instruct-hf"))
for formula in df["0"]:
    output = gpt_inference(formula)
    final_data.append((formula, output, "gpt-4o-mini"))

df_data = pd.DataFrame(final_data, columns=["input", "output", "model"])
# clean codellama outputs
df_data["output"] = df_data["output"].apply(lambda x: [x.split("Answer: ")[1] if "Answer" in x and isinstance(x, str) else x][0])
#strip any whitespace at beginning or end
df_data["output"] = df_data["output"].apply(lambda x: [x.strip() if isinstance(x, str) else x][0])

df_data.to_csv("well_outputs.csv", index=False)
