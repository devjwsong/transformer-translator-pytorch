from tqdm import tqdm

DATA_DIR = '../data'
SRC_RAW_DATA_NAME = 'europarl-v7.fr-en.en'
TAR_RAW_DATA_NAME = 'europarl-v7.fr-en.fr'
SRC_DATA_NAME = 'full_data.en'
TAR_DATA_NAME = 'full_data.fr'

num_samples = 50000


def sampling(file_name):
    texts = []

    with open(f"{DATA_DIR}/{file_name}", 'r') as f:
        lines = f.readlines()

    print(f"Processing {file_name}...")
    for i, line in enumerate(tqdm(lines)):
        if i == num_samples:
            break

        texts.append(line.strip())

    return texts


def make_file(text_list, output_file_name):
    print(f"Making {output_file_name}...")
    with open(f"{DATA_DIR}/{output_file_name}", 'w') as f:
        for text in tqdm(text_list):
            f.write(text + '\n')


if __name__=='__main__':
    eng_texts = sampling(SRC_RAW_DATA_NAME)
    frn_texts = sampling(TAR_RAW_DATA_NAME)

    make_file(eng_texts, SRC_DATA_NAME)
    make_file(frn_texts, TAR_DATA_NAME)
