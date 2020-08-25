"""
Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

from .cnn import DCNN
from .utils import Processor_data, open_txt
from .helpers import find_url, find_emoji, \
                        find_email, find_hash, find_at, find_number, \
                        find_phone_number, find_year, find_nonalp, find_punct, ngrams_top, unique_char, find_coin, \
                        num_great, num_less, find_dates, only_words, boundary, search_string, pick_only_key_sentence, \
                        pick_unique_sentence, find_capital, remove_tag, mac_add, ip_add, subword, lat_lon, pos_look_ahead, \
                        neg_look_ahead, pos_look_behind, neg_look_behind, find_domain


__all__ = ["DCNN", "Processor_data", "open_txt", 
            "find_url", "find_emoji", "find_email",
            "find_hash", "find_at",
            "find_number", "find_phone_number", "find_year",
            "find_nonalp", "find_punct",
            "ngrams_top", "unique_char",
            "find_coin", "num_great", "num_less",
            "find_dates", "only_words", "boundary",
            "search_string", "pick_only_key_sentence",
            "pick_unique_sentence", "find_capital", "remove_tag",
            "mac_add", "ip_add", "subword", "lat_lon", "pos_look_ahead",
            "neg_look_ahead", "pos_look_behind",
            "neg_look_behind", "find_domain"
        ]
