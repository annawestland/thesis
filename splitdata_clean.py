import pandas as pd
import regex as re
import operator as op
import random

def preprocess_formula(formula):
    formula = formula.replace("~", "¬")
    formula = formula.replace("&", "∧")
    formula = formula.replace("|", "∨")
    formula = formula.replace("$", "→")
    formula = formula.replace("@", "∀")
    formula = formula.replace("/", "∃")
    return formula


def connectives(formula):
    """
    Check if a formula contains more than 8 connectives.
    """
    char_list1 = ['∧','∨','¬','→']
    #char_list2 = ['&','|','~','$', '/','@']
    if sum(formula.count(char) for char in (char_list1)) > 8:
        return True

def pijl(string):

    string = string[:string.find(')')]
    pattern = r"^[^()]* → \(?[^()]*\)?$"
    if re.match(pattern,string):
        return True
    
    return False

def implication(formula):
    #remove pattern (a) or (a,b) since all predicates are unary or binary
    pattern = r"\( . \)|\( . , . \)"
    formula = re.sub(pattern, "", formula)
    
    haakjes = ""
    if "→ (" in formula:
        # Extract outer and inner parts
        outer, inner = formula.split("→ (", 1)
        if pijl(inner):
            return True

    return False


def negation(formula):
    """
    Check if a formula has a valid double negation.
    """
    s = "¬ ¬"
    if op.contains(formula,s):
        return True
    
    return False



def split(input):
    """
    split full dataset into:
    nest_list
    con_list
    both_list
    neither_list
    """
    df2 = pd.read_csv(input)
    formula_list = [preprocess_formula(formula) for formula in (df2["formula"].values.tolist())] #length=20187
  
    neg_list = [formula for formula in formula_list if negation(formula)] #length=68
    nest_list = [formula for formula in formula_list if implication(formula)] #length=208
    con_list = [formula for formula in formula_list if connectives(formula)] #length=6861

    length = 68
    #get some formulas out of formula_list that have no overlap
    neither_list = random.sample([x for x in formula_list if x not in con_list and x not in nest_list],1000)

    #get a random sample out of both lists into the 'both_list'
    both_list = random.sample(nest_list,int(length/2)) + random.sample(con_list,int(length/2))
    nest_list = random.sample(nest_list, length)
    con_list = random.sample(con_list,length)
    neg_list = random.sample(neg_list,length)
    
    df_final_data = pd.DataFrame({
    'nested': nest_list,
    'connectives': con_list,
    'negation': neg_list,
    'both': both_list
    })
    df_final_data.to_csv("dataset_sub.csv", index=False)
    df_final_data["nested"] = df_final_data["nested"].apply(lambda x: [x.strip() if isinstance(x, str) else x][0])


    df_final_data = pd.DataFrame(neither_list)
    df_final_data.to_csv("dataset.csv", index=False)
    

    

print(split("clean_dataset.csv"))