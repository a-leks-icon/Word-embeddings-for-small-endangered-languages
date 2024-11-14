#Script written by Aleksandr Schamberger (GitHub: JLEKS)
#Created: 2024-06-05
#Latest Version: 2024-06-07
#The script and its content is licensed under the Creative Commons Attribution ShareAlike (CC BY-SA) 4.0 license.

from corflow import fromElan
import glob
import re

source_data_path = "../data/lang_data/source_data/urum/"
prepared_data_path = "../data/lang_data/prepared_data/"
eaf_files = glob.glob(source_data_path+"*.eaf")

tokens = []
for file in eaf_files:
    file_name = file.replace(source_data_path,"")
    trans = fromElan.fromElan(file,encoding="utf-8")
    #Iterate through all reference tiers in a file.
    for ref_tier in trans.findAllName("ref@"):
        #Get the respective word tier.
        for ch_tier in ref_tier.children():
            if ch_tier.name.startswith("wd@"):
                wd_tier = ch_tier
                break
        #Get the respective morph tier.
        if wd_tier.children():
            if wd_tier.children()[0].name.startswith("mb@"):
                mb_tier = wd_tier.children()[0]
        #Iterate through every reference segment.
        for ref_seg in ref_tier:
            #Skip, if a pause segment appears.
            if ref_seg.content == "<p:>":
                continue
            #The reference segment has to have word segments.
            if wd_tier in ref_seg.childDict():
                for wd_seg in ref_seg.childDict()[wd_tier]:
                    if wd_seg.content.startswith("<"):
                        #Skipt those word segments, which are labelled except for foreign material.
                        if not wd_seg.content.startswith("<<fm"):
                            continue
                    #The word segment has to have morph segments.
                    if wd_seg.childDict()[mb_tier]:
                        for mb_seg in wd_seg.childDict()[mb_tier]:
                            #Remove morph status symbols, dots etc.
                            mb_clean = re.sub(r"[=\-\n\*\’\. ]","",mb_seg.content)
                            if mb_clean == "":
                                continue
                            tokens.append(mb_clean+"\n")
        #Add a dot indicating the end of a sentence after a reference segment.
        tokens.append(".\n")

#Saves all tokens in a new txt file.
with open(prepared_data_path+"urum_morph_tokens.txt", "w") as output:
    output.writelines(tokens)

