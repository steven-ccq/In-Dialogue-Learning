import json
import os, ast

def generate_examples(filepaths):
    """Yields examples."""
    # TODO(cornell_movie_dialog): Yields (key, example) tuples from the dataset
    movie_char_file = os.path.join(filepaths, "movie_characters_metadata.txt")
    movie_conv_file = os.path.join(filepaths, "movie_conversations.txt")
    movie_lines_file = os.path.join(filepaths, "movie_lines.txt")
    movie_titles_file = os.path.join(filepaths, "movie_titles_metadata.txt")

    with open(movie_char_file, "rb") as f:
        movie_char_data = [x.decode("latin").split("+++$+++") for x in f.readlines()]

    with open(movie_conv_file, "rb") as f:
        movie_conv_data = [x.decode("latin").split("+++$+++") for x in f.readlines()]

    with open(movie_lines_file, "rb") as f:
        movie_lines_data = [x.decode("latin").split("+++$+++") for x in f.readlines()]

    with open(movie_titles_file, "rb") as f:
        movie_titles_data = [x.decode("latin").split("+++$+++") for x in f.readlines()]
    # looping over movie conversation file
    for id_, conv in enumerate(movie_conv_data):
        char_id_1 = conv[0]
        char_id_2 = conv[1]
        movie_id = conv[2]
        line_ids = conv[-1].replace("\n", "")
        line_ids = ast.literal_eval(line_ids.strip())
        lines_texts = []
        # searching text corresponding to each lineID in line_ids in movie lines file
        for line_id in line_ids:
            i = 0
            while i < len(movie_lines_data) and movie_lines_data[i][0].strip() != line_id:
                i += 1
            lines_texts.append(movie_lines_data[i][-1].strip())  # if i < len(movie_lines_data) else '')
        # look for char names in movie character file
        j = 0
        while j < len(movie_char_data) and movie_char_data[j][0].strip() != char_id_1.strip():
            j += 1
        char_name_1 = movie_char_data[j][1]  # if j < len(movie_char_data) else ''
        movie_title = movie_char_data[j][3]  # if j < len(movie_char_data) else ''

        k = 0
        while k < len(movie_char_data) and movie_char_data[k][0].strip() != char_id_2.strip():
            k += 1
        char_name_2 = movie_char_data[k][1]

        # look for movie year, IMDBRating, genre, no_imdb_voting in movie tiles file
        li = 0
        while li < len(movie_titles_data) and movie_titles_data[li][0].strip() != movie_id.strip():
            li += 1
        movie_year = movie_titles_data[li][2]
        imdb_rating = movie_titles_data[li][3]
        no_imdb_vote = movie_titles_data[li][4]
        genre = movie_titles_data[li][5].replace("\n", "").strip()
        movie_genres = ast.literal_eval(genre)

        yield id_, {
            "movieID": movie_id,
            "movieTitle": movie_title,
            "movieYear": movie_year,
            "movieIMDBRating": imdb_rating,
            "movieNoIMDBVotes": no_imdb_vote,
            "movieGenres": movie_genres,
            "characterID1": char_id_1,
            "characterID2": char_id_2,
            "characterName1": char_name_1,
            "characterName2": char_name_2,
            "utterance": {"text": lines_texts, "LineID": line_ids},
        }

def get_dataset(input_file_path, output_file_path):
    with open(input_file_path, 'r') as f:
        data_list = [json.loads(_) for _ in f]

    selected_data_list = [_ for _ in data_list if len(_['utterance']['text']) >= 4]

    uid_dials_dict = {}
    for _ in selected_data_list:
        user = _['characterID1'].strip()
        bot = _['characterID2'].strip()
        dial = _['utterance']['text']
        
        if bot not in uid_dials_dict:
            uid_dials_dict[bot] = []
        uid_dials_dict[bot].append(dial)

    for k, v in uid_dials_dict.items():
        id = k
        dials = v
        json_item = {'id': id, 'dials': dials}
        with open(output_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(json_item, ensure_ascii=False))
            f.write('\n')

def get_valid_dataset(input_file_path, output_file_path):
    with open(input_file_path, 'r') as f:
        data_list = [json.loads(_) for _ in f]

    for data in data_list:
        uid = data['id']
        dials = data['dials']
        if len(dials) == 1:
            continue
        for i in range(len(dials)):
            history_dial = dials[:i] + dials[i+1:]
            current_dial = dials[i]
            history_sents = []
            for dial in history_dial:
                if len(dial) % 2 != 0:
                    dial = dial[:-1]
                for sent in dial:
                    history_sents.append(sent)
            if len(current_dial) % 2 != 0:
                current_dial = current_dial[:-1]
            for i in range(0, len(current_dial), 2):
                user = current_dial[i]
                bot = current_dial[i+1]
                history_sents.append(user)
                with open(output_file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'id': uid, 'history': history_sents, 'response': bot}))
                    f.write('\n')
                history_sents.append(bot)

if __name__ == '__main__':
    raw_data_fir = './'
    json_path = './movie-corpus/movie_corpus.json'
    dataset_path = './movie-corpus/movie_IDL.json'
    valid_dataset_path = './movie-corpus/movie_IDL_valid.json'
    dataset = generate_examples(raw_data_fir)
    for data in dataset:
        with open(json_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data[1], ensure_ascii=False))
            f.write('\n')
    get_dataset(json_path, dataset_path)
    get_valid_dataset(dataset_path, valid_dataset_path)