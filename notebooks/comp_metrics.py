from aif360.detectors.mdss.ScoringFunctions.Bernoulli import Bernoulli
from aif360.detectors.mdss.MDSS import MDSS

def lift(data, sub_dict, target_col, direction):
    if direction == 'negative':
        data[target_col] = 1 - data[target_col]
    to_choose = data[sub_dict.keys()].isin(sub_dict).all(axis=1)
    temp_df = data.loc[to_choose]
    return temp_df[target_col].mean()/data[target_col].mean()
    
def support(data, sub_dict):
    to_choose = data[sub_dict.keys()].isin(sub_dict).all(axis=1)
    temp_df = data.loc[to_choose] 
    return len(temp_df)/len(data)
    
def odds_ratio(data, sub_dict, target_col, direction):
    if direction == 'negative':
        data[target_col] = 1 - data[target_col]
    to_choose = data[sub_dict.keys()].isin(sub_dict).all(axis=1)
    in_df =   data.loc[to_choose]
    out_df = data.loc[~to_choose]
    in_odds = in_df[target_col].mean()/(1 - in_df[target_col].mean())
    out_odds = out_df[target_col].mean()/(1 - out_df[target_col].mean())
    odds_ratio = in_odds/out_odds
    return odds_ratio

def mdss_score(data, sub_dict, target_col, direction):
    if direction == 'negative':
        data[target_col] = 1 - data[target_col]
    scoring_function = Bernoulli(direction=direction, alpha = data[target_col].mean())
    scanner = MDSS(scoring_function)
    data['expected'] = data[target_col].mean()
    score = scanner.score_current_subset(coordinates = data, 
                            outcomes = data[target_col], current_subset = sub_dict,
                            expectations = data['expected'], penalty = 0)
    return score

def quality_score(data, sub_dict, target_col,direction):
    if direction == 'negative':
        data[target_col] = 1 - data[target_col]
    to_choose = data[sub_dict.keys()].isin(sub_dict).all(axis=1)
    temp_df = data.loc[to_choose]
    return (len(temp_df)/len(data))  * (temp_df[target_col].mean() - data[target_col].mean())

def get_metrics(method, data, sub_dict, target_col, start, end, direction = 'positive'):
    dff = data.copy()
    if sub_dict:
        l = lift(dff.copy(), sub_dict, target_col, direction)
        s =  support(dff.copy(), sub_dict)
        o = odds_ratio(dff.copy(), sub_dict, target_col, direction)
        m = mdss_score(dff.copy(), sub_dict, target_col, direction)
        q = quality_score(dff.copy(), sub_dict, target_col, direction)
    else:
        l = '-'
        s = '-'
        o = '-'
        m = '-'
        q = '-'
    return [method, sub_dict, l, s, o, m, q, end - start]

def get_metrics_artificial(method, data, sub_dict, start, end, ground_truth):
    dff = data.copy()
    if sub_dict:
        to_choose = dff[sub_dict.keys()].isin(sub_dict).all(axis=1)
        anom_index = set(dff.loc[to_choose].index)
    else:
        anom_index = set(dff.index)

    to_choose = dff[ground_truth.keys()].isin(ground_truth).all(axis=1)
    ground_index = set(dff.loc[to_choose].index)

    intersection = len(ground_index.intersection(anom_index))
    precision = intersection/len(anom_index)
    recall = intersection/len(ground_index)

    return [method, sub_dict, precision, recall, end - start]