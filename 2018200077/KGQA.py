from KG.kg import HarvestText
from rdflib import URIRef, Graph, Namespace, Literal
from pyxdameraulevenshtein import damerau_levenshtein_distance as edit_dis
import numpy as np

class KGQA:
    def __init__(self, SVOs=None, entity_mention_dict=None, entity_type_dict=None):
        self.ht_SVO = HarvestText()
        self.default_namespace = "https://github.com/blmoistawinde/"
        if SVOs:
            self.KG = self.build_KG(SVOs, self.ht_SVO)
        self.ht_e_type = HarvestText()
        self.ht_e_type.add_entities(entity_mention_dict, entity_type_dict)
        self.q_type2templates = {():["default0"],
                                 ("#实体#",):["defaultE"],
                                 ("#谓词#",):["defaultV"],
                                 ("#实体#", "#谓词#"): ["defaultEV"],
                                 ("#实体#", "#实体#"): ["defaultEE"],
                                 ("#谓词#", "#实体#"): ["defaultVE"],}

        self.q_type2search = {():lambda *args: "",
                             ("#实体#",):lambda x: self.get_sparql(x=x),
                             ("#谓词#",):lambda y: self.get_sparql(y=y),
                             ("#实体#", "#谓词#"): lambda x,y: self.get_sparql(x=x, y=y),
                             ("#实体#", "#实体#"): lambda x,z: self.get_sparql(x=x, z=z),
                             ("#谓词#", "#实体#"): lambda y,z: self.get_sparql(y=y, z=z),}
        self.q_template2answer = {"default0":lambda *args: self.get_default_answer(),
                                  "defaultE":lambda entities, answers: self.get_default_answers(entities, answers),
                                  "defaultV": lambda entities, answers: self.get_default_answers(entities, answers),
                                  "defaultEV": lambda entities, answers: self.get_default_answers(entities, answers),
                                  "defaultEE": lambda entities, answers: self.get_default_answers(entities, answers),
                                  "defaultVE": lambda entities, answers: self.get_default_answers(entities, answers),}
    def get_sparql(self,x=None,y=None,z=None,limit=None):
        quest_placeholders = ["", "", "", "", "", ""]
        for i, word in enumerate([x,y,z]):
            if word:
                quest_placeholders[i] = ""
                quest_placeholders[i + 3] = "ns1:"+word
            else:
                quest_placeholders[i] = "?x"+str(i)
                quest_placeholders[i + 3] = "?x"+str(i)

        query0 = """
            PREFIX ns1: <%s> 
            select %s %s %s
            where {
            %s %s %s.
            }
            """ % (self.default_namespace, quest_placeholders[0], quest_placeholders[1], quest_placeholders[2],
                   quest_placeholders[3], quest_placeholders[4], quest_placeholders[5])
        if limit:
            query0 += "LIMIT %d" % limit
        return query0
    def get_default_answer(self,x="",y="",z=""):
        if len(x+y+z) > 0:
            return x+y+z
        else:
            return "换个问题吧！"
    def get_default_answers(self,entities, answers):
        if len(answers) > 0:
            return "、".join("".join(x) for x in answers)
        else:
            return "换个问题吧！"
    def build_KG(self, SVOs, ht_SVO):
        namespace0 = Namespace(self.default_namespace)
        g = Graph()
        type_word_dict = {"实体":set(),"谓词":set()}
        for (s,v,o) in SVOs:
            type_word_dict["实体"].add(s)
            type_word_dict["实体"].add(o)
            type_word_dict["谓词"].add(v)
            g.add((namespace0[s], namespace0[v], namespace0[o]))
        ht_SVO.add_typed_words(type_word_dict)
        return g
    def parse_question_SVO(self,question,pinyin_recheck=False,char_recheck=False):
        entities_info = self.ht_SVO.entity_linking(question,pinyin_recheck,char_recheck)
        entities, SVO_types = [], []
        for span,(x,type0) in entities_info:
            entities.append(x)
            SVO_types.append(type0)
        entities = entities[:2]
        SVO_types = tuple(SVO_types[:2])
        return entities, SVO_types
    def extract_question_e_types(self,question,pinyin_recheck=False,char_recheck=False):
        entities_info = self.ht_e_type.entity_linking(question,pinyin_recheck,char_recheck)
        question2 = self.ht_e_type.decoref(question,entities_info)
        return question2
    def match_template(self,question,templates):
        arr = ((edit_dis(question, template0), template0) for template0 in templates)
        dis, temp = min(arr)
        return temp
    def search_answers(self, search0):
        records = self.KG.query(search0)
        answers = [[str(x)[len(self.default_namespace):] for x in record0] for record0 in records]
        return answers
    def add_template(self, q_type, q_template, answer_function):
        self.q_type2templates[q_type].append(q_template)
        self.q_template2answer[q_template] = answer_function
    def answer(self,question,pinyin_recheck=False,char_recheck=False):
        entities, SVO_types = self.parse_question_SVO(question,pinyin_recheck,char_recheck)
        search0 = self.q_type2search[SVO_types](*entities)
        if len(search0) > 0:
            answers = self.search_answers(search0)
            templates = self.q_type2templates[SVO_types]
            question2 = self.extract_question_e_types(question,pinyin_recheck,char_recheck)
            template0 = self.match_template(question2, templates)
            answer0 = self.q_template2answer[template0](entities,answers)
        else:
            answer0 = self.get_default_answer()
        return answer0
