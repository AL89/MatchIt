from typing import Dict, Literal
import numpy as np
from .padel_match import Match

def match_seeding_points(match:Match, k: int = 32,method:Literal['score','win']='score') -> Dict[str,int]:
    match_seeding_points = {p.name:0 for p in match.team1 + match.team2}
    if not any(match.result):
        return match_seeding_points

    team1_avg = sum(p.seeding_score for p in match.team1) / 2
    team2_avg = sum(p.seeding_score for p in match.team2) / 2

    # Expected score (ELO)
    expected1 = 1 / (1 + 10 ** ((team2_avg - team1_avg) / 400))
    expected2 = 1 - expected1

    if method == 'score':
        # Actual score based on match points
        actual1 = match.result[0] / sum(match.result)
        actual2 = match.result[1] / sum(match.result)
    else: # 'match'
        if match.result[0] > match.result[1]:
            actual1, actual2 = 1, 0
        if match.result[0] < match.result[1]:
            actual1, actual2 = 0, 1
        else:
            actual1 = actual2 = 0.5

    delta1 = np.ceil(k * (actual1 - expected1))
    delta2 = np.ceil(k * (actual2 - expected2))

    for p in match.team1:
        weight = 1 - (match_seeding_points[p.name] - team1_avg) / 400
        # match_seeding_points[p.name] += round(k * (actual1 - expected1))
        match_seeding_points[p.name] += delta1 * weight
    for p in match.team2:
        weight = 1 - (match_seeding_points[p.name] - team2_avg) / 400
        # match_seeding_points[p.name] += round(k * (actual1 - expected1))
        match_seeding_points[p.name] += delta2 * weight
        # match_seeding_points[p.name] += round(k * (actual2 - expected2))
    
    return match_seeding_points
    
    