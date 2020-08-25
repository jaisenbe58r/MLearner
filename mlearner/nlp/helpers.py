"""
Jaime Sendra Berenguer-2020.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import pandas as pd
import re
import emoji
from sklearn.feature_extraction.text import CountVectorizer

"""
Regex Helpers

Major RE functions

re.findall - Module is used to search for “all” occurrences that match a given pattern.
re.sub - Substitute the matched RE patter with given text
re.match - The match function is used to match the RE pattern to string with optional flags
re.search - This method takes a regular expression pattern and a string and searches for that pattern with the string.
We will be mostly using re.findall to detect patterns.

Based in https://www.kaggle.com/raenish/cheatsheet-text-helper-functions
"""


def find_url(string):
    """
    Search URL in text.

    Parameters
    ----------
        string: str
            Text selected to apply transformation

    Examples:
    ---------
    ```python
    sentence="I love spending time at https://www.kaggle.com/"
    find_url(sentence)
    ```
    """
    text = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string)
    return "".join(text)  # converting return value from list to string.


def find_emoji(text):
    """
    Find and convert emoji to text.

    Parameters
    ----------
        text: str
            Text selected to apply transformation

    Examples:
    ---------
    ```python
    sentence="I love () very much ())"
    find_emoji(sentence)
        >>> ['soccer_ball', 'beaming_face_with_smiling_eyes']
    ```
    """
    emo_text = emoji.demojize(text)
    return re.findall(r'\:(.*?)\:', emo_text)


# def remove_emoji(text):
#     """
#     Remove Emoji from text.

#     Parameters
#     ----------
#         text: str
#             Text selected to apply transformation

#     Examples:
#     ---------
#     ```python
#     sentence="Its all about \U0001F600 face"
#     print(sentence)
#     remove_emoji(sentence)
#         >>> Its all about 😀 face
#         >>> 'Its all about  face'
#     ```
#     """
#     emoji_pattern = re.compile("["
#                                 u"\U0001F600-\U0001F64F"  # emoticons
#                                 u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#                                 u"\U0001F680-\U0001F6FF"  # transport & map symbols
#                                 u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#                                 u"\U00002702-\U000027B0"
#                                 u"\U000024C2-\U0001F251"
#                                 "]+", flags=re.UNICODE)
#     return emoji_pattern.sub(r'', text)


def find_email(text):
    """
    Extract email from text.

    Parameters
    ----------
        text: str
            Text selected to apply transformation

    Examples:
    ---------
    ```python
    sentence="My gmail is abc99@gmail.com"
    find_email(sentence)
        >>> 'abc99@gmail.com'
    ```
    """
    line = re.findall(r'[\w\.-]+@[\w\.-]+', str(text))
    return ",".join(line)


def find_hash(text):
    """
    This value is especially to denote trends in twitter.

    Parameters
    ----------
        text: str
            Text selected to apply transformation

    Examples:
    ---------
    ```python
    sentence="#Corona is trending now in the world"
    find_hash(sentence)
        >>> 'Corona'
    ```
    """
    line = re.findall(r'(?<=#)\w+', text)
    return " ".join(line)


def find_at(text):
    """
    @ - Used to mention someone in tweets

    Parameters
    ----------
        text: str
            Text selected to apply transformation

    Examples:
    ---------
    ```python
    sentence="@David,can you help me out"
    find_at(sentence)
        >>> 'David'
    ```
    """
    line = re.findall(r'(?<=@)\w+', text)
    return " ".join(line)


def find_number(text):
    """
    Pick only number from sentence

    Parameters
    ----------
        text: str
            Text selected to apply transformation

    Examples:
    ---------
    ```python
    sentence="2833047 people are affected by corona now"
    find_number(sentence)
        >>> '2833047'
    ```
    """
    line = re.findall(r'[0-9]+', text)
    return " ".join(line)


def find_phone_number(text):
    """
    Spain Mobile numbers have ten digit. I will write that pattern below.

    Parameters
    ----------
        text: str
            Text selected to apply transformation

    Examples:
    ---------
    ```python
    find_phone_number("698887776 is a phone number of Mark from 210,North Avenue")
        >>> '698887776'
    ```
    """
    line = re.findall(r"(\+34|0034|34)?[ -]*(6|7)[ -]*([0-9][ -]*){8}", text)
    return line


def find_year(text):
    """
    Extract year from 1940 till 2040.

    Parameters
    ----------
        text: str
            Text selected to apply transformation

    Examples:
    ---------
    ```python
    sentence="India got independence on 1947."
    find_year(sentence)
        >>> ['1947']
    ```
    """
    line = re.findall(r"\b(19[40][0-9]|20[0-1][0-9]|2040)\b", text)
    return line


def find_nonalp(text):
    """
    Extract Non Alphanumeric characters.

    Parameters
    ----------
        text: str
            Text selected to apply transformation

    Examples:
    ---------
    ```python
    sentence="Twitter has lots of @ and # in posts.(general tweet)"
    find_nonalp(sentence)
        >>> ['@', '#', '.', '(', ')']
    ```
    """
    line = re.findall("[^A-Za-z0-9 ]", text)
    return line


def find_punct(text):
    """
    Retrieve punctuations from sentence.

    Parameters
    ----------
        text: str
            Text selected to apply transformation

    Examples:
    ---------
    ```python
    example="Corona virus have kiled #24506 confirmed cases now.#Corona is un(tolerable)"
    print(find_punct(example))
        >>> ['#', '.', '#', '(', ')']
    ```
    """
    line = re.findall(r'[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]*', text)
    string = "".join(line)
    return list(string)


def ngrams_top(corpus, ngram_range, n=None, idiom='english'):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.

    Examples:
    ---------
    ```python
    ngrams_top(df['text'],(1,1),n=10)
        >>> 	text	count
        >>> 0	just	2278
        >>> 1	day     2115
        >>> 2	good	1578
        >>> 3	like	1353
        >>> 4	http	1247
        >>> 5	work	1150
        >>> 6	today	1147
        >>> 7	love	1145
        >>> 8	going	1103
        >>> 9	got	    1085
    ngrams_top(df['text'],(2,2),n=10)
        >>> 	text	        count
        >>> 0	mother day	    358
        >>> 1	twitpic com	    334
        >>> 2	http twitpic    332
        >>> 3	mothers day	    279
        >>> 4	happy mother    275
        >>> 5	just got	    219
        >>> 6	happy mother    199
        >>> 7	http bit	    180
        >>> 8	bit ly	        180
        >>> 9	good morning    176
    ```
    """
    vec = CountVectorizer(stop_words=idiom, ngram_range=ngram_range).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    total_list = words_freq[:n]
    df = pd.DataFrame(total_list, columns=['text', 'count'])
    return df


def unique_char(sentence):
    """
    Retrieve punctuations from sentence.
    If you want to change match repetitive characters to n numbers,
    chage the return line in the rep function to grp[0:n].

    Parameters
    ----------
        sentence: str
            Text selected to apply transformation

    Examples:
    ---------
    ```python
    sentence="heyyy this is loong textttt sooon"
    unique_char(sentence)
        >>> 'hey this is long text son'
    ```
    """
    def rep(text):
        grp = text.group(0)
        if len(grp) > 1:
            return grp[0:1]  # can change the value here on repetition

    convert = re.sub(r'(\w)\1+', rep, sentence)
    return convert


def find_coin(text, symbol="$"):
    """
    Find prices in text

    Parameters
    ----------
        text: str
            Text selected to apply transformation

        symbol: str
            Coin symbol

    Examples:
    ---------
    ```python
    sentence="this shirt costs $20.56"
    find_dollar(sentence)
        >>> '$20.56'
    ```
    """
    if symbol == "$":
        line = re.findall(r'\$\d+(?:\.\d+)?', text)
    elif symbol == "€":
        line = re.findall(r'\€\d+(?:\.\d+)?', text)
    else:
        raise NameError(f"Coin symbol {symbol} not implemented")
    return " ".join(line)


def num_great(text):
    """
    Number greater than 930

    Parameters
    ----------
        text: str
            Text selected to apply transformation

    Examples:
    ---------
    ```python
    sentence="It is expected to be more than 935 corona death
        and 29974 observation cases across 29 states in india"
    num_great(sentence)
        >>> '935 29974'
    ```
    """
    line = re.findall(r'9[3-9][0-9]|[1-9]\d{3,}', text)
    return " ".join(line)


def num_less(text):
    """
    Number less than 930.

    Parameters
    ----------
        text: str
            Text selected to apply transformation

    Examples:
    ---------
    ```python
    sentence="There are some countries where less than 920 cases
        exist with 1100 observations"
    num_less(sentence)
        >>> '920'
    ```
    """
    only_num = []
    for i in text.split():
        line = re.findall(r'^(9[0-2][0-0]|[1-8][0-9][0-9]|[1-9][0-9]|[0-9])$', i)
        only_num.append(line)
        all_num = [",".join(x) for x in only_num if x != []]
    return " ".join(all_num)


def find_dates(text):
    """
    Find Dates. mm-dd-yyyy format

    Parameters
    ----------
        text: str
            Text selected to apply transformation

    Examples:
    ---------
    ```python
    sentence="Todays date is 04/28/2020 for format mm/dd/yyyy, not 28/04/2020"
    find_dates(sentence)
        >>> [('04', '28', '2020')]
    ```
    """
    line = re.findall(r'\b(1[0-2]|0[1-9])/(3[01]|[12][0-9]|0[1-9])/([0-9]{4})\b', text)
    return line


def only_words(text):
    """
    Only Words - Discard Numbers.

    Parameters
    ----------
        text: str
            Text selected to apply transformation

    Examples:
    ---------
    ```python
    sentence="the world population has grown from 1650 million to 6000 million"
    only_numbers(sentence)
        >>> '1650 6000'
    ```
    """
    line = re.findall(r'\b[^\d\W]+\b', text)
    return " ".join(line)


def boundary(text):
    """
    Extracting word with boundary

    Parameters
    ----------
        text: str
            Text selected to apply transformation

    Examples:
    ---------
    ```python
    sentence="Most tweets are neutral in twitter"
    boundary(sentence)
        >>> 'neutral'
    ```
    """
    line = re.findall(r'\bneutral\b', text)
    return " ".join(line)


def search_string(text, key):
    """
    Is the key word present in the sentence?

    Parameters
    ----------
        text: str
            Text selected to apply transformation

        key: str
            Word to search within the phrase

    Examples:
    ---------
    ```python
    sentence="Happy Mothers day to all Moms"
    search_string(sentence,'day')
        >>> True
    ```
    """
    return bool(re.search(r''+key+'', text))


def pick_only_key_sentence(text, keyword):
    """
    If we want to get all sentence with particular keyword.
    We can use below function.

    Parameters
    ----------
        text: str
            Text selected to apply transformation.

        keyword: str
            Word to search within the phrase.

    Examples:
    ---------
    ```python
    sentence="People are fighting with covid these days.
        Economy has fallen down.How will we survice covid"
    pick_only_key_sentence(sentence,'covid')
        >>> ['People are fighting with covid these days',
            'How will we survice covid']
    ```
    """
    line = re.findall(r'([^.]*'+keyword+'[^.]*)', text)
    return line


def pick_unique_sentence(text):
    """
    Most webscrapped data contains duplicated sentence.
    This function could retrieve unique ones.

    Parameters
    ----------
        text: str
            Text selected to apply transformation.

        keyword: str
            Word to search within the phrase.

    Examples:
    ---------
    ```python
    sentence="I thank doctors\nDoctors are working very hard
        in this pandemic situation\nI thank doctors"
    pick_unique_sentence(sentence)
        >>> ['Doctors are working very hard in this
            pandemic situation', 'I thank doctors']
    ```
    """
    line = re.findall(r'(?sm)(^[^\r\n]+$)(?!.*^\1$)', text)
    return line


def find_capital(text):
    """
    Extract words starting with capital letter.
    Some words like names,place or universal object
    are usually mentioned in a text starting with CAPS.

    Parameters
    ----------
        text: str
            Text selected to apply transformation.

    Examples:
    ---------
    ```python
    sentence="World is affected by corona crisis.
        No one other than God can save us from it"
    find_capital(sentence)
        >>> ['World', 'No', 'God']
    ```
    """
    line = re.findall(r'\b[A-Z]\w+', text)
    return line


def remove_tag(string):
    """
    Most of web scrapped data contains html tags.
    It can be removed from below re script

    Parameters
    ----------
        text: str
            Text selected to apply transformation.

    Examples:
    ---------
    ```python
    sentence="Markdown sentences can use <br> for breaks and <i></i> for italics"
    remove_tag(sentence)
        >>> 'Markdown sentences can use  for breaks and  for italics'
    ```
    """
    text = re.sub('<.*?>', '', string)
    return text


def mac_add(string):
    """
    Extract Mac address from text.
    https://stackoverflow.com/questions/26891833/python-regex-extract-mac-addresses-from-string/2689237

    Parameters
    ----------
        string: str
            Text selected to apply transformation.

    Examples:
    ---------
    ```python
    sentence="MAC ADDRESSES of this laptop - 00:24:17:b1:cc:cc .
        Other details will be mentioned"
    mac_add(sentence)
        >>> ['00:24:17:b1:cc:cc']
    ```
    """
    text = re.findall('(?:[0-9a-fA-F]:?){12}', string)
    return text


def ip_add(string):
    """
    Extract IP address from text.

    Parameters
    ----------
        string: str
            Text selected to apply transformation.

    Examples:
    ---------
    ```python
    sentence="An example of ip address is 125.16.100.1"
    ip_add(sentence)
        >>> ['125.16.100.1']
    ```
    """
    text = re.findall('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', string)
    return text


def subword(string, sub):
    """
    Extract number of subwords from sentences and words.

    Parameters
    ----------
        string: str
            Text selected to apply transformation.

        sub: str
            subwords from sentences

    Examples:
    ---------
    ```python
    sentence = 'Fundamentalism and constructivism are important skills'
    subword(sentence,'ism') # change subword and try for others
        >>> 2
    ```
    """
    text = re.findall(sub, string)
    return len(text)


def lat_lon(string, display=False):
    """
    valid latitude & longitude

    Parameters
    ----------
        string: str
            Text selected to apply transformation.

    Examples:
    ---------
    ```python
    lat_lon('28.6466772,76.8130649')
    lat_lon('2324.3244,3423.432423')
        >>> [28.6466772,76.8130649] is valid latitude & longitude
        >>> [2324.3244,3423.432423] is not a valid latitude & longitude
    ```
    """
    text = re.findall(r'^[-+]?([1-8]?\d(\.\d+)?|90(\.0+)?),\s*[-+]?(180(\.0+)?|((1[0-7]\d)|([1-9]?\d))(\.\d+)?)$', string)
    if text != []:
        if display:
            print("[{}] is valid latitude & longitude".format(string))
        return True
    else:
        if display:
            print("[{}] is not a valid latitude & longitude".format(string))
        return False


def pos_look_ahead(string, A, B):
    """
    Positive look ahead will succeed if passed non-consuming expression
    does match against the forthcoming input.
    The syntax is A(?=B) where A is actual expression and B
    is non-consuming expression.

    Parameters
    ----------
        string: str
            Text selected to apply transformation.

        A, B: str
            A is actual expression and B is non-consuming expression.

    Examples:
    ---------
    ```python
    pos_look_ahead("I love kaggle. I love DL","love","DL")
        >>> position:(17, 21) Matched word:love
    ```
    """
    pattern = re.compile(''+A+'(?=\s'+B+')')
    match = pattern.search(string)
    print("position:{} Matched word:{}".format(match.span(), match.group()))


def neg_look_ahead(string, A, B):
    """
    Negative look ahead will succeed if passed non-consuming expression
    does not match against the forthcoming input.
    The syntax is A(?!B) where A is actual expression and B
    is non-consuming expression.

    Parameters
    ----------
        string: str
            Text selected to apply transformation.

        A, B: str
            A is actual expression and B is non-consuming expression.

    Examples:
    ---------
    ```python
    neg_look_ahead("I love kaggle. I love DL","love","DL")
        >>> position:(2, 6) Matched word:love
    ```
    """
    pattern = re.compile(''+A+'(?!\s'+B+')')
    match = pattern.search(string)
    print("position:{} Matched word:{}".format(match.span(), match.group()))


def pos_look_behind(string, A, B):
    """
    Positive look behind will succeed if passed non-consuming expression
    does match against the forthcoming input.
    The syntax is A(?<=B) where A is actual expression and B
    is non-consuming expression.

    Parameters
    ----------
        string: str
            Text selected to apply transformation.

        A, B: str
            A is actual expression and B is non-consuming expression.

    Examples:
    ---------
    ```python
    pos_look_behind("i love nlp.everyone likes nlp","love","nlp")
    # the word "nlp" that do come after "love"
        >>> position:(7, 10) Matched word: nlp
    ```
    """
    pattern = re.compile("(?<="+A+"\s)"+B+"")
    match = pattern.search(string)
    print("position:{} Matched word: {}".format(match.span(), match.group()))


def neg_look_behind(string, A, B):
    """
    Positive look behind will succeed if passed non-consuming expression
    does not match against the forthcoming input.
    The syntax is "A(?<!=B)" where "A"is actual expression and "B"
    is non-consuming expression.

    Parameters
    ----------
        string: str
            Text selected to apply transformation.

        A, B: str
            A is actual expression and B is non-consuming expression.

    Examples:
    ---------
    ```python
    neg_look_behind("i love nlp.everyone likes nlp","love","nlp")
    # the word "nlp" that doesnt come after "love"
        >>> position:(26, 29) Matched word: nlp
    ```
    """
    pattern = re.compile("(?<!"+A+"\s)"+B+"")
    match = pattern.search(string)
    print("position:{} Matched word: {}".format(match.span(), match.group()))


def find_domain(string):
    """
    Search domains in the text.

    Parameters
    ----------
        string: str
            Text selected to apply transformation.

        A, B: str
            A is actual expression and B is non-consuming expression.

    Examples:
    ---------
    ```python
    sentence="WHO provides valid information about covid in their site who.int.
        UNICEF supports disadvantageous childrens. know more in unicef.org"
    find_domain(sentence)
        >>> ['who.int', 'unicef.org']
    ```
    """
    text = re.findall(r'\b(\w+[.]\w+)', string)
    return text
