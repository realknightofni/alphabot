from fuzzywuzzy import process

def match_names(clean_names, dirty_names):
    matched_names = []
    
    for clean_name in clean_names:
        # Find the closest match in the dirty_names list
        match, score = process.extractOne(clean_name, dirty_names)
        matched_names.append((clean_name, match, score))
    
    return matched_names
