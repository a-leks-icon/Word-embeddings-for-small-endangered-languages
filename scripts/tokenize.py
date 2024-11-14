#Script written by Aleksandr Schamberger (GitHub: JLEKS)
#Created: 2024-06-29
#Latest Version: 2024-06-29
#The script and its content is licensed under the Creative Commons Attribution ShareAlike (CC BY-SA) 4.0 license.

import re

input_path = "../data/lang_data/source_data/"
output_path = "../data/lang_data/prepared_data/"
file_name = "angela_merkel.txt"

with open(input_path+file_name, "r") as file:
    text = file.readlines()
text = " ".join(text)
text = re.findall("[\u00c4\u00e4\u00d6\u00f6\u00dc\u00fc\u00dfa-zA-Z]+",text)
text2 = []
for word in text:
    text2.append(word+"\n")
text = text2
with open(output_path+file_name, "w") as file:
    file.writelines(text)