import re


class TokenPOSGrouper:
    def __init__(self):
        self.tokens_by_pos = {}
        self.already_seen = []

    def add(self, token, tk_txt):
        pos = token.pos_
        lemma = token.lemma_
        is_tag = re.match(r"<.*?>", tk_txt)
        is_punct = tk_txt in ['.', ',', '?', '!', '/', ';', ':', '`', '\n']

        if tk_txt in self.already_seen:
            return

        if self._can_add_tag(is_tag, tk_txt):
            self.tokens_by_pos["TAG"]["TAG"].append(tk_txt)
            self.already_seen.append(tk_txt)
            return

        if self._can_add_propn(pos, tk_txt):
            self.tokens_by_pos[pos][pos].append(tk_txt)
            self.already_seen.append(tk_txt)
            return

        if self._can_add_noun(pos, tk_txt):
            self.tokens_by_pos[pos][pos].append(tk_txt)
            self.already_seen.append(tk_txt)
            return

        if self._can_add_punct(is_punct, tk_txt):
            self.tokens_by_pos["PUNCT"]["PUNCT"].append(tk_txt)
            self.already_seen.append(tk_txt)
            return

        if self._can_add_lemma(pos, lemma, tk_txt):
            self.tokens_by_pos[lemma][pos].append(tk_txt)
            self.already_seen.append(tk_txt)

    def _can_add_tag(self, is_tag, tk_txt):
        if is_tag and "TAG" not in self.tokens_by_pos:
            self.tokens_by_pos["TAG"] = {
                "TAG": []
            }
            return True

        if is_tag and "TAG" in self.tokens_by_pos:
            if tk_txt not in self.tokens_by_pos["TAG"]["TAG"]:
                return True

        return False

    def _can_add_propn(self, pos, tk_txt):
        if pos == "PROPN" and pos not in self.tokens_by_pos:
            self.tokens_by_pos[pos] = {
                pos: []
            }
            return True

        if pos == "PROPN" and pos in self.tokens_by_pos:
            if tk_txt not in self.tokens_by_pos[pos][pos]:
                return True

        return False

    def _can_add_noun(self, pos, tk_txt):
        if pos == "NOUN" and pos not in self.tokens_by_pos:
            self.tokens_by_pos[pos] = {
                pos: []
            }
            return True
        elif pos == "NOUN" and pos in self.tokens_by_pos:
            if tk_txt not in self.tokens_by_pos[pos][pos]:
                return True

        return False

    def _can_add_punct(self, is_punct, tk_txt):
        if is_punct and "PUNCT" not in self.tokens_by_pos:
            self.tokens_by_pos["PUNCT"] = {
                "PUNCT": []
            }
            return True
        elif is_punct and "PUNCT" in self.tokens_by_pos:
            if tk_txt not in self.tokens_by_pos["PUNCT"]["PUNCT"]:
                return True

        return False

    def _can_add_lemma(self, pos, lemma, tk_txt):
        if lemma not in self.tokens_by_pos:
            self.tokens_by_pos[lemma] = {
                pos: []
            }
            return True
        elif lemma in self.tokens_by_pos and pos not in self.tokens_by_pos[lemma]:
            self.tokens_by_pos[lemma][pos] = []
            return True
        elif lemma in self.tokens_by_pos and pos in self.tokens_by_pos[lemma]:
            if tk_txt not in self.tokens_by_pos[lemma][pos]:
                return True

        return False
