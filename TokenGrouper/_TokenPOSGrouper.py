import re


class TokenPOSGrouper:
    def __init__(self):
        self.tokens_by_pos = {}
        self.already_seen = []

    def add(self, token, tk_txt):
        pos = token.pos_
        lemma = token.lemma_
        is_tag = re.match(r"<.*?>", tk_txt)

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

        if pos not in self.tokens_by_pos:
            self.tokens_by_pos[pos] = {
                lemma: [tk_txt]
            }
            self.already_seen.append(tk_txt)
        elif pos in self.tokens_by_pos and lemma not in self.tokens_by_pos[pos]:
            self.tokens_by_pos[pos][lemma] = [tk_txt]
            self.already_seen.append(tk_txt)
        elif pos in self.tokens_by_pos and lemma in self.tokens_by_pos[pos]:
            if tk_txt not in self.tokens_by_pos[pos][lemma]:
                self.tokens_by_pos[pos][lemma].append(tk_txt)
                self.already_seen.append(tk_txt)
