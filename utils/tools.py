import numpy as np

MAX_SEN_LEN = 50


def construct_dict(path, word_dict, label_dict, words_in_given_embedding):
    """
    word_dict
    label_dict
    doc format:
    w1 w2 w3 ... \t label
    """
    err = 0
    word_set = set()
    label_set = set()
    with open(path) as fin:
        for line in fin:
            if not line.strip():
                continue
            if len(line.strip().split("\t")) != 2:
                err += 1
                continue
            word_set |= {x.strip() for x in line.strip().split("\t")[0].split()}
            label_set.add(line.strip().split("\t")[1].strip())
    print("err_instance: %d" % err)
    word_set |= set(words_in_given_embedding)
    with open(word_dict, "w") as fout:
        for word in word_set:
            fout.writelines(word + "\n")
        fout.writelines("OTHER-WORDS-ID")
    with open(label_dict, "w") as fout:
        for label in label_set:
            fout.writelines(label + "\n")


def load_dict(path):
    """
    word_dict
    label_dict
    entype_dict
    ret: {word1:id1, word2:id2,...}
    """
    ret = {}
    with open(path) as fin:
        for idx, line in enumerate(fin):
            if not line.strip():
                continue
            ret[line.strip()] = idx + 1  # id from 1, 0 is for empty word/label/entype
    return ret


def load_given_embeddings(path):
    if path is None:
        return None
    ans = {}
    with open(path, "rb") as fin:
        for line in fin:
            tmps = line.strip().split()
            if len(tmps) < 3:
                continue
            ans[tmps[0].strip()] = [float(x) for x in tmps[1:]]
    return ans


def _load_random_embeddings(dim, word_dict_p):
    word_dict = load_dict(word_dict_p)
    larggest_id = word_dict["OTHER-WORDS-ID"]
    rng = np.random.RandomState(3135)
    return rng.uniform(low=-0.5, high=0.5, size=(larggest_id + 1, dim))


def _word_init_embeddings(dest_p, word_dict_p, dim=50, given_embs=None):
    word_dict = load_dict(word_dict_p)
    larggest_id = word_dict["OTHER-WORDS-ID"]
    rng = np.random.RandomState(3135)
    unkown_words = list(rng.uniform(low=-0.5, high=0.5, size=(dim,)))
    cnt = 0
    if given_embs is None:
        embs = rng.uniform(low=-0.5, high=0.5, size=(larggest_id + 1, dim))
    else:
        miss_words = [x for x in word_dict if x not in given_embs]

        re_dict = {v: k for k, v in word_dict.items()}
        embs = []
        embs.append(list(np.zeros(dim)))  # empty word embedding
        for idx in range(1, larggest_id):  # note that the largest id is not included
            word = re_dict[idx]
            if word in given_embs:
                embs.append(list(given_embs[word]))
            else:
                embs.append(list(rng.uniform(low=-0.5, high=0.5, size=(dim,))))
        embs.append(
            list(rng.uniform(low=-0.5, high=0.5, size=(dim,)))
        )  # other word embedding
        print("total words: %d, missing words: %d" % (len(word_dict), len(miss_words)))
    print("errors", cnt)
    with open(dest_p, "w") as fout:
        for line in embs:
            fout.writelines(" ".join(map(str, line)) + "\n")


def init_embeddings(position_dim, word_dim, given_embs, area):

    word_dest_p = "data/%s/embeddings/word/%d.txt" % (area, word_dim)
    word_dict_p = "data/%s/dicts/word_dict.txt" % area
    _word_init_embeddings(word_dest_p, word_dict_p, word_dim, given_embs)


def load_embedding(path):
    """
    load word embedding or
    position embedsing or
    label embedding
    """
    if path is None:
        return _load_random_embeddings(100, "data/dicts/word_dict.txt")
    fin = open(path)
    data = []
    for line in fin:
        if not line.strip():
            continue
        data.append([float(x) for x in line.strip().split(" ")])
    return np.array(data, dtype="float32")
