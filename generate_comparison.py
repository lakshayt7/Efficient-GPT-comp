import pandas as pd 
from openai import OpenAI
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--ifile', dest='ifile', type=str, default='')        #input-file
parser.add_argument('--ofile', dest='ofile', type=str, default='')        #input-file

args = parser.parse_args()


client = OpenAI()

#COT prompt
prompt = '''Your job is to compare two different financial documents. The first text contains a rule from the first document, and the second text contains relevant texts from the second document. 
First, highlight 3 points of similarity between the texts. Next, highlight three points of difference between the texts. Finally, return a concluding statement that gives a summary of the comparison between the two texts. Do not list the difference as the shorter text having less detail than the longer text specify differences in application.


Think step by step and slowly and highlight intermediate reasoning steps. Give references to specific points when giving differences and avoid giving generic differences'''

df = pd.read_csv(args.ifile)
indices = df['Index Doc1'].unique()[1:]

gpt_responses =  pd.DataFrame(columns=["Text 1", "Text 2", "Prompt", "Response"])
for indice in indices:
    #find matching texts from both documents and create the full prompt
    df_i = df[df['Index Doc1'] == indice]['Sentence Doc2']
    matched_text = '\n'.join(df_i.to_list())
    text = df[df['Index Doc1'] == indice]['Sentence Doc1'].to_list()[0]
    full_prompt = f"{prompt}\nText 1:\n{text}\nText 2:\n{matched_text}"

    #generate completion using gpt-4
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a financial expert. Your job is to compare two financial texts and highlight similarities and differences between them."},
            {"role": "user", "content": full_prompt}
        ]
    )


    #add response to dataframe
    gpt_responses = pd.concat([gpt_responses, pd.DataFrame([{
            "Text 1": text,
            "Text 2": matched_text,
            "Prompt": full_prompt,
            "Response": completion.choices[0].message
        }])], ignore_index=True)

#get only the completion not other stuff

gpt_responses['Response'] = gpt_responses['Response'].apply(lambda x : x.content)#[31:-57].replace("\\n", "\n"))
gpt_responses['print'] = gpt_responses['Text 1'] + "\n"+ gpt_responses['Text 2'] + "\n\n" + gpt_responses['Response']
prints = gpt_responses['print'].to_list()

final_o =  "\n\n".join(prints)

#now given the individual comparisons generate the final comparison
fprompt= '''Given the text below which list simiarities and differences between different rules. Take all the similarities and create a five point summary of similarities with references to the rule they come from.
 Now go over the differences section carefully and create a five point summary of differences between them along with a reference to the key consideration they come from.
'''

final_prompt = f"{fprompt}\n\n{final_o}"

completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a financial expert. Your job is to compare two financial texts and highlight similarities and differences between them."},
            {"role": "user", "content": final_prompt}
        ]
    )
print(completion[0].message)

#save csv
gpt_responses.to_csv(args.ofile, index=False)
