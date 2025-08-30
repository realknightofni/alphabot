import argparse
import fnmatch
import os
import shutil
import time
from collections import Counter, defaultdict

import cv2
import easyocr
import imagehash
import pandas as pd
from matchparse import artifacts, genres, hashes, heroes, placements, traits
from matchparse.base import add_text_top_left, draw_bboxes, draw_contours
from PIL import Image

INPUT_DIR = 'input'
OUTPUT_DIR = 'output/'

LANGUAGES = ['ch_tra', 'en']
IMG_EXTENSION_PATTERNS = {'*.jpg', '*.png'}

# for QA only
COLORS = [
    (0, 128, 0),      # Green
    (26, 140, 0), (51, 153, 0), (77, 166, 0), (102, 179, 0), (128, 191, 0),
    (153, 204, 0), (179, 217, 0), (204, 230, 0), (230, 242, 0),    # Yellow-green
    (255, 255, 0),    # Yellow
    (255, 230, 0), (255, 204, 0), (255, 179, 0), (255, 153, 0), (255, 128, 0),
    (255, 102, 0),(255, 77, 0),(255, 51, 0),(255, 26, 0),
    (255, 0, 0),      # Red
    (230, 0, 0),(204, 0, 0),(179, 0, 0),(153, 0, 0),
    (180, 105, 255),  # PINK - only used for unknowns
]
WORST_COLOR_INDEX = len(COLORS) - 2

def get_color_from_hamming(hamming_distance):
    if hamming_distance == 9999 or hamming_distance is None:
        color_index = -1
    else:
        color_index = min(hamming_distance, WORST_COLOR_INDEX)
    
    return COLORS[color_index]


def is_within_bbox(y_center, bbox):
    x, y, w, h = bbox
    return y <= y_center <= y+h


class Player:

    def __init__(self, name, placement, y_center):
        self.name = name
        self.placement = placement
        self.y_center = y_center

        self.artifact_bboxes = []
        self.artifact_hashes = []
        self.artifact_guesses = []
        self.artifact_hammings = []
        self.unknown_artifact_indexes = []

        self.hero_bbox = None
        self.hero_hash = None
        self.hero_guess = None
        self.hero_hamming = None

        self.trait_bboxes = []
        self.trait_hashes = []
        self.trait_guesses = []
        self.trait_hammings = []
        self.unknown_trait_indexes = []

        self.genre_bboxes = []
        self.genre_hashes = []
        self.genre_guesses = []
        self.genre_hammings = []
        self.initial_genre_levels = []
        self.initial_genre_exps = []
        self.genre_levels = []
        self.genre_exps = []

        self.genre_star_contours = []
        self.genre_lvl_changes = []
        self.genre_exp_changes = []

    def get_guess_scores(self):
        num_artifacts = len(self.artifact_bboxes)
        num_heroes = 1 if self.hero_bbox else 0
        num_traits = len(self.trait_hashes)

        score_artifacts = sum([1 if hashes.is_unknown(a) else 0 for a in self.artifact_guesses])
        score_heroes = 1 if not hashes.is_unknown(self.hero_bbox) else 0
        score_traits = sum([1 if hashes.is_unknown(t) else 0 for t in self.trait_guesses])

        return ((score_artifacts, num_artifacts),
                (score_heroes, num_heroes),
                (score_traits, num_traits)
                )

    def order_bboxes(self):
        # NOTE: might want to handle cases where we only detect 2 artifact bboxes
        self.artifact_bboxes = sorted(self.artifact_bboxes, key=lambda b: b[0])  # sort by x
        self.trait_bboxes = sorted(self.trait_bboxes, key=lambda b: b[0])  # sort by x

    def add_artifact_bbox(self, bbox):
        self.artifact_bboxes.append(bbox)

    def add_hero_bbox(self, bbox):
        if self.hero_bbox is not None:
            raise ValueError(f'Player {self.name} already has a hero bbox!')
        self.hero_bbox = bbox

    def add_trait_bbox(self, bbox):
        self.trait_bboxes.append(bbox)

    def guess_icons(self, image, reader):
        if self.hero_bbox:
            self.hero_guess, self.hero_hash, self.hero_hamming = hashes.guess_hero_hash(image, self.hero_bbox)

        self.order_bboxes()
        for i, bbox in enumerate(self.artifact_bboxes):
            artifact_guess, artifact_hash, artifact_hamming = hashes.guess_artifact_hash(image, bbox)
            self.artifact_guesses.append(artifact_guess)
            self.artifact_hashes.append(artifact_hash)
            self.artifact_hammings.append(artifact_hamming)

            if hashes.is_unknown(artifact_guess):
                self.unknown_artifact_indexes.append(i)

        trait_hero = None if hashes.is_unknown(self.hero_guess) else self.hero_guess.name
        for i, bbox in enumerate(self.trait_bboxes):
            # TODO: might want to add some logic to account for non-dupes.
            # i.e. you cannot get the same trait twice
            primary_color, _ = traits.determine_primary_bbox_color(image, bbox)
            trait_guess, trait_hash, trait_hamming = hashes.guess_trait_hash(image, bbox,
                                                                             hero=trait_hero, color=primary_color)
            self.trait_guesses.append(trait_guess)
            self.trait_hashes.append(trait_hash)
            self.trait_hammings.append(trait_hamming)

            if hashes.is_unknown(trait_guess):
                self.unknown_trait_indexes.append(i)

        if hashes.is_unknown(self.hero_guess):
            counts = Counter([t.hero_name for t in self.trait_guesses if hashes.is_known(t)])
            more_than_threshold = any(cnt > 2 for cnt in counts.values())

            total_sum = sum(counts.values())
            common = counts.most_common(1)
            most_common_hero, most_common_cnt = common[0] if common else '', 0
            is_majority = most_common_cnt >= (total_sum - most_common_cnt)

            if more_than_threshold and is_majority:
                print(f'Inferring hero from traits : {most_common_hero} - hash: {self.hero_hash}')
                self.hero_guess = hashes.Hero(most_common_hero, '_')
            else:
                self.hero_guess = hashes.UNKNOWN_HERO
                #raise ValueError(f'Could not infer hero from traits: {[t.name for t in self.trait_guesses]}')
        
        for i, bbox in enumerate(self.genre_bboxes):
            genre_guess, genre_hash, genre_hamming = hashes.guess_genre_hash(image, bbox)
            genre_level, star_contours = genres.get_genre_level(image, bbox)
            genre_exp, _ = genres.read_genre_exp(image, bbox, reader)

            self.genre_guesses.append(genre_guess)
            self.genre_hashes.append(genre_hash)
            self.genre_hammings.append(genre_hamming)
            self.initial_genre_levels.append(genre_level)
            self.initial_genre_exps.append(genre_exp)

            self.genre_star_contours.extend(star_contours)

    def save_icons(self, image, output_dir):
        def _save_icons(bbox, icon_type, is_unknown=False):
            if bbox is None:
                return

            x, y, w, h = bbox
            icon = image[y : y + h, x : x + w]
            image_hash = hashes.get_icon_hash(image, bbox)

            suffix = "_u.png" if is_unknown else ".png"
            output_fn = f"{image_hash}_{icon_type}_{w}x{h}{suffix}"
            output_path = os.path.join(output_dir, icon_type)
            os.makedirs(output_path, exist_ok=True)
            output_fp = os.path.join(output_path, output_fn)
            cv2.imwrite(output_fp, icon)

        is_hero_unknown = hashes.is_unknown(self.hero_guess)
        _save_icons(self.hero_bbox, "heroes", is_hero_unknown)

        for i, bbox in enumerate(self.artifact_bboxes):
            is_unknown = i in self.unknown_artifact_indexes
            _save_icons(bbox, "artifacts", is_unknown)

        for i, bbox in enumerate(self.trait_bboxes):
            is_unknown = i in self.unknown_trait_indexes
            _save_icons(bbox, "traits", is_unknown)
        
        for i, bbox in enumerate(self.genre_bboxes):
            _save_icons(bbox, "genres")
    

    # TODO: remove at some point or refactor. currently used in a utility script
    def save_unknown_trait_icons(self, image, output_dir):
        is_hero_known = hashes.is_known(self.hero_guess)
        if not is_hero_known:
            return

        hero_name = self.hero_guess.name

        cnt = 0
        for i, bbox in enumerate(self.trait_bboxes):
            is_unknown = i in self.unknown_trait_indexes
            if not is_unknown:
                continue
            if bbox is None:
                continue

            x, y, w, h = bbox
            icon = image[y : y + h, x : x + w]
            image_hash = hashes.get_icon_hash(image, bbox)

            primary_color, _ = traits.determine_primary_bbox_color(icon)

            output_fn = f"{image_hash}_{primary_color}_{hero_name.lower()}_{w}x{h}.png"
            output_fp = os.path.join(output_dir, 'traits', output_fn)
            cv2.imwrite(output_fp, icon)
            cnt += 1

        print(f'Saved {cnt} unknown traits')


class MatchParser:
    
    def __init__(self, image_filepath):
        self.image_filepath = str(image_filepath)
        self.image = cv2.imread(self.image_filepath)
        self.image_height, self.image_width = self.image.shape[:2]

        self.reader = easyocr.Reader(LANGUAGES)

        self.artifact_bboxes = []
        self.hero_bboxes = []
        self.trait_bboxes = []
        self.missing_trait_bboxes = []

        self.players = []

        self.reporter_name = None

        self.genres_main = []
        self.genres_banned = []

    def run(self, output_dir=OUTPUT_DIR, is_save_icons=True):
        try:
            self.main()
        except Exception:
            # TODO: expand on this
            print('Error occured')
            raise

        self.save_processed_image(output_dir)
        print('Saved processed image')

        if is_save_icons:
            print('Saving icons...')
            self.save_icons(output_dir)

    def main(self):
        print(f'reading player placements: {self.image_filepath}')
        mid_x = self.image_width // 2
        t0 = time.time()
        self.ocr_results = self.reader.readtext(self.image[:,:mid_x],
                                                width_ths=1.5,  # merge close bboxes
                                                height_ths=0.7)
        t1 = time.time()
        print(f'TIME DELTA {t1} - {t0} = {t1-t0}')
        self.player_placements = placements.get_player_placements(self.ocr_results)
        placement_string = '\n'.join(f'{p[0]} - {p[1]} ({p[2]:.3f})  ({p[3]})' for p in self.player_placements)
        print(placement_string)

        self.reporter_placement, self.reporter_name = placements.get_reporter_placement(self.player_placements, self.image)

        self.players = [Player(p[1], p[0], p[3]) for p in self.player_placements]   # TODO: align the player placements w/ the class
        self.players = sorted(self.players, key=lambda p: p.placement)

        print('Getting bboxes')
        self.artifact_bboxes = artifacts.get_artifact_bboxes(self.image)
        print(f'{len(self.artifact_bboxes)} artifact bboxes')

        max_x = max([p[4][2][0] for p in self.player_placements if p[4]])
        self.hero_bboxes = heroes.get_hero_bboxes(self.image, max_x=max_x)
        print(f'{len(self.hero_bboxes)} hero bboxes')

        self.trait_bboxes, self.missing_trait_bboxes = traits.get_trait_and_missing_bboxes(self.image)
        print(f'{len(self.trait_bboxes)} trait bboxes')

        reference_width, base_x = genres.calculate_reference_width(self.artifact_bboxes)
        self.associate_bboxes(reference_width, base_x)
        for player in self.players:
            player.guess_icons(self.image, self.reader)

        self.genres_main = genres.infer_main_genres(self.players)
        self.genres_banned = genres.get_banned_genres(self.genres_main)

        for player in self.players:
            player.genre_guesses = genres.fix_genre_guesses(player, self.genres_main)
            player.genre_levels, player.genre_lvl_changes = genres.fix_genre_levels(player.initial_genre_levels,
                                                                                    player.initial_genre_exps)
            player.genre_exps, player.genre_exp_changes = genres.fix_genre_exps(player.initial_genre_exps,
                                                                                player.genre_levels,
                                                                                player.genre_guesses)

        text = self.to_text()
        print(text)
        # jsn = self.to_json()
        # print(jsn)
        rows, headers = self.to_rows()
        df = pd.DataFrame(rows, columns=headers)
        df.to_csv('output.tsv', sep='\t', index=False)
        print('saved tsv')
    
    def score_main(self):
        num_players = len(self.players)

        hero_unknowns, _ = hashes.count_known_vs_total([p.hero_guess for p in self.players])

        artifact_unknown = 0
        trait_unknown, trait_total = 0, 0
        genre_unknown = 0
        for player in self.players:
            _au, _ = hashes.count_known_vs_total(player.artifact_guesses)
            artifact_unknown += _au

            _tu, _tt = hashes.count_known_vs_total(player.trait_guesses)
            trait_unknown += _tu
            trait_total += _tt

            _gu, _ = hashes.count_known_vs_total(player.genre_guesses)
            genre_unknown += _gu

        scores = {'num_heroes_unknown': hero_unknowns,
                  'num_heroes': num_players,
                  'num_artifact_unknown': artifact_unknown,
                  'num_artifacts': artifacts.EXPECTED_NUM_ARTIFACTS,
                  'num_trait_unknown': trait_unknown,
                  'num_traits': trait_total,
                  'num_genres_unknown': genre_unknown,
                  'num_genres': num_players * 8,
                  }
        return scores

    def save_icons(self, output_dir):
        if output_dir:
            for player in self.players:
                player.save_icons(self.image, output_dir)
            print('Saved icons')

    def associate_bboxes(self, reference_width, base_x):
        ycen2player = {p.y_center: p for p in self.players}
        y_centers = sorted(ycen2player.keys())
        lost_found = defaultdict(list)

        for bbox in self.artifact_bboxes:
            is_associated = False
            for y_center in y_centers:
                if is_within_bbox(y_center, bbox):
                    ycen2player[y_center].add_artifact_bbox(bbox)
                    is_associated = True

            if not is_associated:
                lost_found['artifact'].append(bbox)
        
        for player in self.players:
            player.genre_bboxes = genres.infer_genre_bboxes(player.artifact_bboxes,
                                                            reference_width,
                                                            base_x)

        # print('Hero bboxes:')
        # print([(y, y+h) for x,y,w,h in self.hero_bboxes])
        for bbox in self.hero_bboxes:
            is_associated = False
            for y_center in y_centers:
                if is_within_bbox(y_center, bbox):
                    ycen2player[y_center].add_hero_bbox(bbox)
                    is_associated = True

            if not is_associated:
                lost_found['hero'].append(bbox)

        for bbox in self.trait_bboxes:
            is_associated = False
            for y_center in y_centers:
                if is_within_bbox(y_center, bbox):
                    ycen2player[y_center].add_trait_bbox(bbox)
                    is_associated = True

            if not is_associated:
                lost_found['trait'].append(bbox)

        print(f'Lost found: {lost_found}')
        if lost_found:
            lf_cnts = {k:len(v) for k, v in lost_found.items()}
            total_cnts = len(self.artifact_bboxes) + len(self.hero_bboxes) + len(self.trait_bboxes)
            print(f'Found some bboxes that could not be associated {lf_cnts} / {total_cnts}')
            # raise ValueError(f'Could not associate bboxes: {lost_found}')

    def to_text(self):
        player_names = [p.name for p in self.players]
        text = f'Players: {player_names}\n'
        text += f'Genres: {self.genres_main} (Banned: {self.genres_banned})'

        for player in self.players:
            is_reporter = player.name == self.reporter_name
            reporter_text = '*' if is_reporter else ''
            text += f'\n#{player.placement} - {player.name}{reporter_text} - {player.hero_guess.name} - {player.hero_hash}'

            artifact_names = [guess.name for guess in player.artifact_guesses]
            text += f'\n\tArtifacts: {artifact_names}'
            text += f'\n\t\tHashes: {[str(h) for h in player.artifact_hashes]}'
            
            trait_names = [guess.name for guess in player.trait_guesses]
            text += f'\n\tTraits: {trait_names}'
            text += f'\n\t\tHashes: {[str(h) for h in player.trait_hashes]}'

            unique_genre_names = set(guess.name for guess in player.genre_guesses if guess.name != 'UNKNOWN_GENRE')
            genre_codes = [genres.to_genre_shorthand(guess.name, lvl, exp) for guess, lvl, exp in zip(player.genre_guesses,
                                                                                                 player.genre_levels,
                                                                                                 player.genre_exps)]
            text += f'\n\t{len(unique_genre_names)} Genres: {genre_codes}'
            text += f'\n\tHashes: {[str(h) for h in player.genre_hashes]}'

        return text

    def to_rows(self):
        genre_lvl_headers = [f'{g.lower()}_lvl' for g in genres.GENRES]
        genre_exp_headers = [f'{g.lower()}_exp' for g in genres.GENRES]
        headers = ['placement', 'player', 'hero',
                   'trait_1', 'trait_2', 'trait_3', 'trait_4', 'trait_5', 'trait_6',
                   'artifact_1', 'artifact_2', 'artifact_3',
                   'is_reporter', 'reporter_name',
                   'ban1', 'ban2', 'ban3', 'ban4',
                   *genre_lvl_headers,
                   *genre_exp_headers,
                   ]
        rows = []
        for player in self.players:
            is_reporter = player.name == self.reporter_name
            trait_names = [t.name for t in player.trait_guesses]
            trait6 = trait_names + [''] * (6-len(trait_names))

            artifact_names = [a.name for a in player.artifact_guesses]
            artifact3 = artifact_names + [''] * (3-len(artifact_names))
            
            if self.genres_banned:
                banned_genres = sorted(self.genres_banned)[:4]   # TODO: we should put a check somewhere else if the genre ban list > 4
            else:
                banned_genres = ['', '', '', '']
            genre_levels = genres.to_genre_lvl_row(player)
            genre_exps = genres.to_genre_exp_row(player)

            row = (player.placement, player.name, player.hero_guess.name,
                   *trait6,
                   *artifact3,
                    'True' if is_reporter else '', self.reporter_name,
                    *banned_genres,
                    *genre_levels,
                    *genre_exps,
                   )
            rows.append(row)

        return rows, headers

    def to_tabbed(self):
        rows, headers = self.to_rows()

        text = '\t'.join(headers)
        text += '\n'
        for row in rows:
            text += '\t'.join(str(x) for x in row)
            text += '\n'

        return text

    def to_df(self):
        rows, headers = self.to_rows()
        df = pd.DataFrame(rows, columns=headers)
        return df

    def to_json(self):
        players_data = []
        for player in self.players:
            _data = {'name': player.name,
                     'placement': player.placement,
                     'hero': player.hero_guess.name,
                     'traits': [guess.name for guess in player.trait_guesses],
                     'artifacts': [guess.name for guess in player.artifact_guesses],
                     'genres': [(guess.name, lvl, exp) for guess, lvl, exp in zip(player.genre_guesses,
                                                                                  player.genre_levels,
                                                                                  player.genre_exps)]}
            players_data.append(_data)

        data = {'game_duration': '00:00',
                'game_complete_confidence': 0,
                'players': [player.name for player in self.players],
                'genres': self.genres_main,
                'genres_banned': self.genres_banned,
                'map': 'UNKNOWN',
                'submitter': self.reporter_name if self.reporter_name else 'UNKNOWN',
                'region': 'UNKNOWN',
                'game_type': 'UNKNOWN',
                'info': {'players': players_data},
                'data_version': "1"
                }

        return data

    def to_json_debug(self):
        players_data = []
        for player in self.players:
            _data = {'name': player.name,
                     'placement': player.placement,
                     'hero': player.hero_guess.name,
                     'traits': [guess.name for guess in player.trait_guesses],
                     'artifacts': [guess.name for guess in player.artifact_guesses],
                     'y_center': player.y_center,
                     'artifact_hashes': [h for h in player.artifact_hashes],
                     'hero_hash': player.hero_hash,
                     'trait_hashes': [h for h in player.trait_hashes],}
            players_data.append(_data)

        data = {'game_duration': '00:00',
                'game_complete_confidence': 0,
                'players': [],
                'map': 'UNKNOWN',
                'submitter': 'UNKNOWN',
                'region': 'UNKNOWN',
                'game_type': 'UNKNOWN',
                'info': {'players': players_data},
                'data_version': "1"
                }

        return data

    def save_processed_image(self, output_dir):
        image_copy = self.image.copy()

        # draw a color scale "legend"
        numbers = [0, 5, 10, 15, 20, 24]
        for i, num in enumerate(numbers):
            x = 400 + i*35
            y = 20
            color = COLORS[num]
            add_text_top_left(image_copy, f'{num:02}',
                              font_scale=0.8, thickness=2,
                              position=(x, y), rgb_color=color)

        for player in self.players:
            if player.hero_bbox:
                color = get_color_from_hamming(player.hero_hamming)
                draw_bboxes(image_copy, [player.hero_bbox], color)

                x, y, w, h = player.hero_bbox
                text = player.hero_guess.name
                add_text_top_left(image_copy, text,
                                  font_scale=0.5,
                                  thickness=1,
                                  position=(x, y-10),
                                  rgb_color=color,
                                  )

            for i, bbox in enumerate(player.artifact_bboxes):
                color = get_color_from_hamming(player.artifact_hammings[i])
                draw_bboxes(image_copy, [bbox], color)

                artifact_guess = player.artifact_guesses[i]
                # text = shorten_words(artifact_guess.name)
                text = artifact_guess.name[:5]
                x, y, w, h = bbox
                add_text_top_left(image_copy, text,
                                  font_scale=0.5,
                                  thickness=1,
                                  position=(x+4, y-10),
                                  rgb_color=color,
                                  )

            for i, bbox in enumerate(player.trait_bboxes):
                color = get_color_from_hamming(player.trait_hammings[i])
                draw_bboxes(image_copy, [bbox], color)

                trait_guess = player.trait_guesses[i]
                # text = shorten_words(trait_guess.name)
                text = trait_guess.name[:4]
                x, y, w, h = bbox
                add_text_top_left(image_copy, text,
                                  font_scale=0.5,
                                  thickness=1,
                                  position=(x+2, y-12),
                                  rgb_color=color,
                                  )

            # draw_bboxes(image_copy, player.genre_bboxes, (255, 255, 255), thickness=1)
            draw_contours(image_copy, player.genre_star_contours, (0, 244, 207), draw_bbox=False)
            for i, bbox in enumerate(player.genre_bboxes):
                color = get_color_from_hamming(player.genre_hammings[i])
                draw_bboxes(image_copy, [bbox], color, thickness=1)

                genre_name = player.genre_guesses[i].name
                genre_level = player.genre_levels[i]
                genre_exp = player.genre_exps[i]
                genre_sh = genres.to_genre_shorthand(genre_name, genre_level, genre_exp)
                exp_min, exp_max = genres.get_exp_range_from_level(genre_level)

                is_missing_data = '??' in genre_sh or '-1' in genre_sh
                is_match = exp_min <= genre_exp <= exp_max
                is_lvl_agree = genres.agrees_with_neighors(genre_level, i, player.genre_levels)
                is_exp_agree = genres.agrees_with_neighors(genre_exp, i, player.genre_exps)
                is_lvl_change = player.genre_lvl_changes[i]
                is_exp_change = player.genre_exp_changes[i]
                if is_missing_data:
                    color = (255, 0, 0)
                elif not is_match:
                    color = COLORS[-1]
                elif not is_lvl_agree or not is_exp_agree:
                    color = (255, 64, 0)
                elif is_lvl_change:
                    color = (255, 128, 0)
                elif is_exp_change:
                    color = (255, 255, 0)
                else:
                    color = (0, 244, 207)

                x, y, w, h = bbox
                add_text_top_left(image_copy, genre_sh,
                                  font_scale=0.4,
                                  thickness=1,
                                  position=(x+5, y-10),
                                  rgb_color=color)

                if is_exp_change:
                    text = f'E{player.initial_genre_exps[i]:02d}'
                    add_text_top_left(image_copy, text,
                                      font_scale=0.4,
                                      thickness=1,
                                      position=(x+w-35, y+h//2),
                                      rgb_color=color)

                if is_lvl_change:
                    text = f'\nL{player.initial_genre_levels[i]:02d}'
                    add_text_top_left(image_copy, text,
                                      font_scale=0.5,
                                      thickness=1,
                                      position=(x+w-35, y+h//2),
                                      rgb_color=color)

        # draw missing trait boxes
        draw_bboxes(image_copy, self.missing_trait_bboxes, (255, 255, 255), thickness=1)

        # Add placement text to the top left
        top_left = [bbox[0] for _, _, _, _, bbox in self.player_placements]
        start_x = min(x for x,y in top_left)

        for placement, player_name, conf, y_cen, _ in self.player_placements:
            text = f'{placement} - {player_name}'
            add_text_top_left(image_copy, text, font_scale=0.5, thickness=1, position=(start_x, y_cen-30))

        image_fn = os.path.basename(self.image_filepath)
        output_fp = os.path.join(output_dir, f'output_{image_fn}')
        cv2.imwrite(output_fp, image_copy)
        print(f'Saved to: {output_fp}')
