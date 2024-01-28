import pandas as pd
import re

DESC_STR = "c_charge_desc"

with open("c_charge_desc_categs.txt") as fp: categs = fp.read().split('\n\n')

replacements = dict()
for categ in categs:
	vals = categ.split('\n')
	label = vals[0].replace(':', '')
	for i in range(1, len(vals)):
		vals[i] = vals[i].replace('"', '')
		if not (vals[i] in replacements.keys()):
			replacements[vals[i]] = label
		
df = pd.read_csv("compas_data.csv")

descs = []
for index, row in df.iterrows():
	descs.append(row[DESC_STR])

with open("compas_data.csv") as fp: data = fp.readlines()

print(data[0], end="")
for i in range(0, len(descs)):
	if descs[i] in replacements.keys():
		data[i + 1] = data[i + 1].replace(descs[i], replacements[descs[i]])

	
	print(re.sub(' +', ' ', data[i + 1]), end="")