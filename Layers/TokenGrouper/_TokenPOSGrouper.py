import re


class TokenPOSGrouper:
    def __init__(self):
        self.tokens_by_pos = {}
        self.already_seen = []

    def add(self, token, tk_txt):
        pos = token.pos_
        lemma = token.lemma_
        is_tag = re.match(r"<.*?>", tk_txt)
        is_punct = tk_txt in ['.', ',', '?', '!', '/', ';', ':']

        if tk_txt in self.already_seen:
            return

        if is_tag and "TAG" not in self.tokens_by_pos:
            self.tokens_by_pos["TAG"] = {
                "TAG": [tk_txt]
            }
            self.already_seen.append(tk_txt)
            return
        if is_tag and "TAG" in self.tokens_by_pos:
            if tk_txt not in self.tokens_by_pos["TAG"]["TAG"]:
                self.tokens_by_pos["TAG"]["TAG"].append(tk_txt)
                self.already_seen.append(tk_txt)

            return

        if pos == "PROPN" and pos not in self.tokens_by_pos:
            self.tokens_by_pos[pos] = {
                pos: [tk_txt]
            }
            self.already_seen.append(tk_txt)
        elif pos == "PROPN" and pos in self.tokens_by_pos:
            if tk_txt not in self.tokens_by_pos[pos][pos]:
                self.tokens_by_pos[pos][pos].append(tk_txt)
                self.already_seen.append(tk_txt)
        elif pos == "NOUN" and pos not in self.tokens_by_pos:
            self.tokens_by_pos[pos] = {
                pos: [tk_txt]
            }
            self.already_seen.append(tk_txt)
        elif pos == "NOUN" and pos in self.tokens_by_pos:
            if tk_txt not in self.tokens_by_pos[pos][pos]:
                self.tokens_by_pos[pos][pos].append(tk_txt)
                self.already_seen.append(tk_txt)
        elif is_punct and "PUNCT" not in self.tokens_by_pos:
            self.tokens_by_pos["PUNCT"] = {
                "PUNCT": [tk_txt]
            }
            self.already_seen.append(tk_txt)
        elif is_punct and "PUNCT" in self.tokens_by_pos:
            if tk_txt not in self.tokens_by_pos["PUNCT"]["PUNCT"]:
                self.tokens_by_pos["PUNCT"]["PUNCT"].append(tk_txt)
                self.already_seen.append(tk_txt)
        elif lemma not in self.tokens_by_pos:
            self.tokens_by_pos[lemma] = {
                pos: [tk_txt]
            }
            self.already_seen.append(tk_txt)
        elif lemma in self.tokens_by_pos and pos not in self.tokens_by_pos[lemma]:
            self.tokens_by_pos[lemma][pos] = [tk_txt]
            self.already_seen.append(tk_txt)
        elif lemma in self.tokens_by_pos and pos in self.tokens_by_pos[lemma]:
            if tk_txt not in self.tokens_by_pos[lemma][pos]:
                self.tokens_by_pos[lemma][pos].append(tk_txt)
                self.already_seen.append(tk_txt)
