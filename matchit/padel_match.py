from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
from uuid import UUID, uuid4
from typing import List, Optional, Union, Tuple, Literal, Dict
from functools import cached_property
import pandas as pd
import random
from .padel_player import Player

class Match(BaseModel,validate_assignment=True):
    id: UUID = Field(default_factory=uuid4)
    team1: List[Player] = Field(min_length=2,max_length=2)
    team2: List[Player] = Field(min_length=2,max_length=2)
    team1_score: int = 0
    team2_score: int = 0
    court_name: Optional[str] = None

    @computed_field
    def winner(self) -> Optional[List[Player]]:
        if self.team1_score > self.team2_score:
            winner_team = self.team1
        elif self.team2_score > self.team1_score:
            winner_team = self.team2
        else:
            winner_team = None
        return winner_team
    
    @computed_field
    def loser(self) -> Union[List[Player],None]:
        winner_team = self.winner
        # if winner is None:
        #     return None
        if not winner_team:
            return False
        return self.team1 if winner_team == self.team2 else self.team2

    @computed_field
    def result(self) -> Tuple[int]:
        return (self.team1_score,self.team2_score)

    def svg_match_court(self):
        from .draw_padel_court import svg_padel_court
        return svg_padel_court(team1=[p.name for p in self.team1],team2=[p.name for p in self.team2],team1_score=self.team1_score,team2_score=self.team2_score,court_name=self.court_name)

    def __repr__(self):
        repr_str = [
            f"{' and '.join([p.name for p in self.team1])}: {self.team1_score}",
            f"{' and '.join([p.name for p in self.team2])}: {self.team2_score}",
        ]
        return "\n".join(repr_str)
    

    @classmethod
    def from_player_list(cls,player_list:List[Player],round_no:int=1,**kwargs):
        return cls(
            team1 = player_list[:2],team2=player_list[2:],
            tournament_name=kwargs.get('tournament_name',None),game_type=kwargs.get('game_type',None),padel_round=round_no,
            court_name=kwargs.get('court_name',None)
        )

class Round(BaseModel,validate_assignment=True):
    id: UUID = Field(default_factory=uuid4)
    round_no: int = 1
    matches: List[Match]
    sitovers: Optional[List[Player]] = None

    @computed_field
    @cached_property
    def player_list(self) -> List[Player]:
        players = []
        for m in self.matches:
            players += m.team1 + m.team2
        return players
    
    @computed_field
    @cached_property
    def team_pairings(self) -> List[Tuple[Player]]:
        players = self.player_list.copy()
        pairings = []
        for i in range(1,len(players),2):
            pair = (players[i-1],players[i])
            pairings.append(pair)
        return pairings

    def is_pair(self,team_pair:Tuple[Player]) -> bool:
        for pair in self.team_pairings:
            if pair[0] in team_pair and pair[1] in team_pair:
                return True
        return False
        # return team_pair in self.team_pairings

    def is_combination(self,combination:Tuple[Player]):
        pair1 = combination[:2]
        pair2 = combination[2:]
        all_combs = [self.is_pair(pair1),self.is_pair(pair2)]
        # all_combs = [self.player_list[i]==combination[i] for i in range(len(combination))]
        return any(all_combs)

    @classmethod
    def from_player_list(cls,player_list:List[Player],round_no:int=1,**kwargs):
        if len(player_list) < 4:
            raise ValueError('Cannot create matches. There must be at least 4 players')
        
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
        
        player_list = list(chunks(player_list,4))
        matches = []
        sit_overs = None
        for m in player_list:
            if len(m) < 4:
                sit_overs = m
                break
            
            matches.append(
                Match(
                    team1 = m[:2],team2 = m[2:],
                    tournament_name=kwargs.get('tournament_name','Padel Event'),
                    game_type=kwargs.get('game_type','Match'),
                    padel_round=round_no
                )
            )
        return cls(round_no=round_no,matches=matches,sit_overs=sit_overs)

    def __eq__(self, other:"Round") -> bool:
        if len(self.player_list) != len(other.player_list):
            return False
        elif not self.sitovers and not other.sitovers:
            return self.player_list == other.player_list
        sitover_bool = any([p in other.sitovers for p in self.sitovers])
        return sitover_bool

    def __str__(self):
        return f"Round {self.round_no}"

    def __repr__(self):
        repr_str = [str(self)]
        repr_str.append(f"Matches: {len(self.matches)}")
        if self.sit_overs:
            repr_str(f"Sit overs: {len(self.sit_overs)}")
        return "\n".join(repr_str) 
    
# def new_padel_round(player_list:List[Player],round_no:int=1,**kwargs) -> Round:
#     total_players = len(player_list)
#     overflow = total_players % 4
        
#     sitover_idx_start = (round_no - 1) * overflow % total_players if overflow else 0
#     sitovers = []

#     if overflow:
#         sitovers = player_list[sitover_idx_start:sitover_idx_start+overflow]
#         if len(sitovers) < overflow:
#             sitovers += player_list[0:overflow - len(sitovers)]
    
#     # Remaining players for match generation
#     active_players = [p for p in player_list if p not in sitovers]
#     matches = []

#     for i in range(0,len(active_players),4):
#         match_players = active_players[i:i+4]
#         matches.append(
#             Match.from_player_list(
#                 player_list=match_players,round_no=round_no,tournament_name=kwargs.get('tournament_name','Padel Event'),game_type=kwargs.get('game_type','Padel')
#             )
#         )
    
#     return Round(round_no=round_no,matches=matches,sitovers=sitovers)

class Event(BaseModel):
    event_name: str
    event_type: Literal['Random','Americano','Mexicano'] = 'Random'
    round: int = 0
    rounds: Union[List[Round],List] = []
    play_by: Literal['points','win'] = 'points'
    player_list: List[Player] = Field(min_length=4)

    @computed_field
    def max_rounds(self) -> int:
        return len(self.rounds) - 1

    @computed_field
    def current_round(self) -> Optional[Round]:
        return self.rounds[-1] if self.rounds else None
    
    @model_validator(mode='before')
    @classmethod
    def validate_event_type(cls,v:dict) -> "Event":
        if v['event_type'] in ('Americano','Mexicano'):
            if not len(v['player_list']) % 4 == 0:
                raise ValueError('An Americano/Mexicano tournament event must have the number of players divisible by 4.')
        return v
    
    # @computed_field
    def standings(self,sort_by:Literal['points','win']='points',return_type:str='dataframe') -> Union[pd.DataFrame,Dict[str,Dict]]:
        if not self.rounds:
            return None
        sort_by_options = ['points','win','draw','loss']
        sort_by = [sort_by] + [sb for sb in sort_by_options if sb != sort_by]
        self.update_player_scores()
        # self.update_player_list(method=sort_by)
        player_standings = [player.model_dump() for player in self.player_list]
        df = pd.DataFrame.from_records(player_standings).sort_values(by=sort_by,ascending=False)
        df = df.rename(columns={c:c.capitalize() for c in df.columns})
        df = df.set_index('Name')
        if return_type == 'dataframe':
            return df
        else:   # dict
            return df.to_dict('index')
        # return player_standings
    
    def update_player_scores(self):
        for player in self.player_list:
            player.points = 0
            player.win = 0
            player.loss = 0
            player.draw = 0

        for round in self.rounds:
            for m in round.matches:
                for player in m.team1:
                    player.points += m.team1_score
                for player in m.team2:
                    player.points += m.team2_score
                
                if not m.winner:
                    for player in m.team1 + m.team2:
                        player.draw += 1
                else:
                    for player in m.winner:
                        player.win += 1
                    for player in m.loser:
                        player.loss += 1

    def update_player_list(self,method:Literal['round','points','seeding_score','win']='round'):
        all_players = self.current_round.player_list.copy() if self.rounds else self.player_list.copy()
        if method in ('win','points','seeding_score'):
            all_players.sort(key=lambda p: getattr(p,method),reverse=True)
            # self.player_list = self.current_round.player_list if self.rounds else self.player_list
        # elif method == 'points':
        #     all_players.sort(key=lambda p: getattr(p,'points'),reverse=True)
        #     # self.player_list = self.update_player_list(method='round')
        #     # self.player_list.sort(key=lambda p: getattr(p,'points'),reverse=True)
        # elif method == 'seeding':
        #     all_players.sort(key=lambda p: getattr(p,'seeding_score'),reverse=True)
        self.players = {p.id:p for p in all_players}

    def randomize_new_round(self,**kwargs) -> Round:
        players = self.player_list
        random.shuffle(players)
        random_round = Round.from_player_list(players,round_no=self.round,**kwargs)
        return random_round
    
    def next_round(self,update_players=True,method:Literal['round','points','seeding_score','win']='round',**kwargs):
        if self.event_type == 'Mexicano':   # Pre-settings for Mexicano
            update_players = True
            method = 'points'
        if update_players:
            self.update_player_scores()
            self.update_player_list(method=method)
        
        self.round = len(self.rounds) + 1
        if self.event_type == 'Random':
            new_round = self.randomize_new_round(**kwargs)
            while new_round in self.rounds:
                new_round = self.randomize_new_round(**kwargs)
        else:
            matches = []
            player_list = self.player_list.copy()
            while len(player_list) >= 4:
                combination = player_list[:4]
                if self.rounds and self.event_type == 'Americano':
                    randomize_count = 0
                    while any([r.is_combination(combination) for r in self.rounds]):
                        random.shuffle(player_list)
                        combination = [player_list[0]] + player_list[2:4] + [player_list[1]]

                        if self.max_rounds + 1 <= self.round: # len(self.players):
                            randomize_count += 1
                        
                        if randomize_count == 3:       # 
                            break

                    team1 = combination[:2]
                    team2 = combination[2:]
                else: # Mexicano
                    # Form teams: Best + worst vs 2nd + 3rd
                    team1 = [combination[0],combination[-1]]
                    team2 = [combination[1],combination[2]]

                matches.append(
                    Match(padel_round=self.round,team1=team1,team2=team2,tournament_name=self.name,game_type=self.event_type)
                )
                for p in combination:
                    player_list.remove(p)

            new_round = Round(round_no=self.round,matches=matches)
        self.rounds.append(new_round)
        self.round += 1
    
    def create_n_rounds(self,n_rounds=4,**kwargs):
        for i in range(n_rounds):
            self.next_round(update_players=False,**kwargs)

# class Americano(Event):
#     @field_validator('event_type')
#     @classmethod
#     def set_event_type(cls,v:str):
#         v = 'Americano'
#         return v
#     @field_validator('player_list')
#     @classmethod
#     def validate_player_count(cls,v:List[Player]) -> List[Player]:
#         if not len(v) % 4 == 0:
#             raise ValueError('An Americano tournament event must have the number of players divisible by 4.')
#         return v

#     def next_round(self,**kwargs):
#         self.update_player_scores()
#         self.update_player_list(method='round')
#         player_list = self.player_list.copy()
#         if len(player_list) % 4 == 0:
#             raise ValueError('An Americano event must have the number of players divisible by 4. Add more players (>= 4) and try again...')

#         round_player_list = []
#         self.round = len(self.rounds) + 1

#         matches = []
#         while len(player_list) >= 4:
#             combination = [player_list[0]] + player_list[2:4] + [player_list[1]]
#             if self.rounds:
#                 randomize_count = 0
#                 while any([r.is_combination(combination) for r in self.rounds]):
#                     random.shuffle(player_list)
#                     combination = [player_list[0]] + player_list[2:4] + [player_list[1]]

#                     if self.max_rounds + 1 <= self.round: # len(self.players):
#                         randomize_count += 1
                    
#                     if randomize_count == 3:       # 
#                         break

#             team1 = combination[:2]
#             team2 = combination[2:]
#             matches.append(
#                 Match(padel_round=self.round,team1=team1,team2=team2,tournament_name=self.name,game_type=self.game_type)
#                         # Match.from_player_list(player_list=list(combination),round_no=self.round,tournament_name=self.name,game_type='Americano')
#             )
#             for p in combination:
#                 player_list.remove(p)
#                 round_player_list.append(p)

#         self.rounds.append(
#             Round(round_no=self.round,matches=matches)
#         )
#         # self.update_player_list(method='round')

# class Mexicano(Event):
#     @field_validator('player_list')
#     @classmethod
#     def validate_player_count(cls,v:List[Player]) -> List[Player]:
#         if not len(v) % 4 == 0:
#             raise ValueError('A Mexicano tournament event must have the number of players divisible by 4.')
#         return v

#     @field_validator('event_type')
#     @classmethod
#     def set_event_type(cls,v:str):
#         v = 'Mexicano'
#         return v

#     def next_round(self,method:Literal['points','win']='points',**kwargs):
#         self.update_player_scores()
#         self.update_player_list(method=method)
#         self.round = len(self.rounds) + 1

#         matches = []
#         for i in range(0,len(self.player_list),4):
#             group = self.player_list[i:i+4]

#             # Form teams: Best + worst vs 2nd + 3rd
#             team1 = [group[0],group[-1]]
#             team2 = [group[1],group[2]]

#             matches.append(
#                 Match(padel_round=self.round,team1=team1,team2=team2,tournament_name=self.name,game_type=self.event_type)
#                         # Match.from_player_list(player_list=list(combination),round_no=self.round,tournament_name=self.name,game_type='Americano')
#             )

#         self.rounds.append(
#             Round(round_no=self.round,matches=matches)
#         )