from pydantic import BaseModel, Field, model_validator, computed_field
from typing import Union, List, Dict, Literal
import pandas as pd
from .padel_player import Player
from .padel_match import Match
from .padel_algorithms import match_seeding_points

def get_player_rankings() -> Union[List,List[Player]]:
    import os
    if not os.path.isfile('player_rankings.xlsx'):
        return []
    
    df = pd.read_excel('player_rankings.xlsx',sheet_name=0)
    players = [Player(name=player['name'],seeding_score=player['score']) for _, player in df.iterrows()]
    return players

def ranking_player_list(player_names:List[str]) -> List[Player]:
    players = get_player_rankings()
    for player in player_names:
        if player not in players:           # Hvis ny spiller
            new_player = Player(name=player)
            players.append(new_player)
    return players

class GroupRankingMatches(BaseModel):
    player_list: List[Player] = Field(min_length=4,max_length=4)
    matches: List[Match]
    k: int = 32

    @model_validator(mode='before')
    @classmethod
    def init_group_matches(cls,v:dict) -> "GroupRankingMatches":
        v['matches'] = []

    # def __init__(self):
        # Round 1
        team1 = [v['player_list'][0],v['player_list'][3]]
        team2 = [v['player_list'][1],v['player_list'][2]]
        v['matches'].append(
            Match(padel_round=1,team1=team1,team2=team2,tournament_name=v.get('event_name','Padel Ranking Games'),game_type='Rankings')
        )

        # Round 2
        team1 = [v['player_list'][0],v['player_list'][2]]
        team2 = [v['player_list'][1],v['player_list'][3]]
        v['matches'].append(
            Match(padel_round=2,team1=team1,team2=team2,tournament_name=v.get('event_name','Padel Ranking Games'),game_type='Rankings')
        )

        # Round 3
        team1 = [v['player_list'][0],v['player_list'][1]]
        team2 = [v['player_list'][2],v['player_list'][3]]
        v['matches'].append(
            Match(padel_round=3,team1=team1,team2=team2,tournament_name=v.get('event_name','Padel Ranking Games'),game_type='Rankings')
        )

        # Delete values if any
        if 'event_name' in v:
            del v['event_name']
        return v

class RankingGames(BaseModel):
    event_name: str = 'Padel Rank Games'
    player_list: List[Player] = Field(min_length=4)
    groups: List[GroupRankingMatches]
    score_by: Literal['score','win'] = 'score'
    k: int = 32
    
    @model_validator(mode='before')
    @classmethod
    def init_groups(cls,v:dict) -> "RankingGames":
        if not len(v['player_list']) % 4 == 0:
            raise ValueError('The number of participants must be divisible by 4.')
        
        if not 'groups' in v:
            v['groups'] = [
                GroupRankingMatches(
                    player_list=v['player_list'][i:i+4],event_name=v['event_name']) for i in range(0,len(v['player_list']),4
                )
            ]
        return v
    
    @computed_field
    def max_n_matches(self) -> int:
        return max([len(group.matches) for group in self.groups])

    @computed_field
    def seeding_scores(self) -> List[Dict[str,int]]:
        match_seeding_scores = {p.name:0 for p in self.player_list}
        match_seeding_scores = [match_seeding_scores.copy() for _ in range(self.max_n_matches)]
        player_seeding_scores = [p.model_dump(exclude=['win','loss','draw']) for p in self.player_list]
        for group in self.groups:
            for i,match in enumerate(group.matches):
                match_seeding_scores = match_seeding_points(match=match,k=self.k,method=self.score_by)
                if not any(match_seeding_scores.values()):
                    continue
                for player,score in match_seeding_scores.items():
                    player_values = [p for p in player_seeding_scores if p['name']==player][0]
                    player_values[f'Match {i+1}'] = score
                    # match_seeding_scores[i][player] += score
        
        return player_seeding_scores
    
    def df_results_table(self) -> pd.DataFrame:
        from .helper_funcs import df_cell_color

        df = pd.DataFrame.from_records(self.seeding_scores).rename(columns={'name':'Name','seeding_score':'Score'}).set_index('Name')
        match_columns = df.filter(like='Match ').columns.to_list()

        df['Total +/-'] = df[match_columns].sum(axis=1)
        df['New score'] = df['Before'] + df['Total +/-']

        df[match_columns] = df[match_columns + ['Total +/-']].style.map(df_cell_color)
        return df