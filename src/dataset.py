import os, sys, re
import numpy as np
from mylib.texthelper import dataset
from mylib.texthelper import myprint as mp
from mylib.texthelper import format
import xml.dom.minidom as xmldom
from lxml import etree, html

class Instance:
    def __init__(self):
        self.entities = []
        self.relations = []
        self.title = ''
        self.p = ''
        self.annolist = []
    def addEntity(self, d):
        self.entities.append(d)

    def addRela(self, d):
        self.relations.append(d)

    def addAnnoTextList(self, l, s):
        self.annolist.append((s,l))

class Token:
    def __init__(self, token, label):
        self.text = token.text
        self.label = label
        self.pos = token.pos_
        self.tag = token.tag_
        self.shape = token.shape_
        self.is_alpha = token.is_alpha
        self.is_stop = token.is_stop
    def __str__(self):
        return '\t'.join(['%s:%s' % item for item in self.__dict__.items()])

class Dataset:
    def __init__(self, dataPath):
        self.dataPath = dataPath
        import spacy
        from collections import defaultdict
        self.nlp = spacy.load('en')

    def xmlParse(self, xmlfile, instance):
        parser = etree.HTMLParser(encoding='utf-8')
        html = etree.parse(xmlfile,parser)
        title = html.xpath('//h1')[0].text.strip()
        title = re.sub('\r\n',' ',title)
        p = html.xpath('//p')[0].text.strip()
        p = re.sub('\r\n',' ',p)
        instance.title = title
        instance.p = p

    def annoParse(self, txtfile, instance):
        with open(txtfile, 'r', encoding="utf-8") as f:
            text = f.read()
            regex=re.compile("\[\d\][\s\S]*?(?:\n\n|$)")
            units = regex.findall(text)
            for unit in units:
                d = dict()
                for item in unit[3:].strip().split('\n'):
                    regx = re.compile('([^=]*) = (.*)')
                    key = regx.search(item).group(1)
                    value = regx.search(item).group(2)
                    d[key] = value
                    # print(key)
                # print(d['type'])
                if d['type']=='"span"':
                    instance.addEntity(d)
                elif d['type']=='"relation"':
                    instance.addRela(d)

    def dataIter(self):
        for d in np.array(sorted(os.listdir(self.dataPath))).reshape(-1,2):
            annofile = self.dataPath+d[0]
            xmlfile = self.dataPath+d[1]
            instance = Instance()
            self.xmlParse(xmlfile,instance)
            self.annoParse(annofile,instance)
            yield instance

    def nerAnnoIter(self):
        for ins in self.dataIter():
            annoString = ins.p
            ins_s = ins.p
            Material_poss = []
            Value_poss = []
            # print(ins.entities)
            for i in ins.entities:
                position = i['position'].split(',')
                bpos = int(position[0][1:])-len(ins.title)-5
                epos = int(position[1][:-1])-len(ins.title)-5
                assert(ins.p[bpos:epos]==i['text'][1:-1])
                if "Material" in i['label']:
                    Material_poss.append((bpos,epos))
                elif i['label'] == '"Value"':
                    Value_poss.append((bpos,epos))
                else:
                    print(i['label'],(bpos,epos))
            Material_poss.sort()
            Value_poss.sort()

            doc = self.nlp(ins.p)
            p_attributed = []
            for token in doc:
                # print(token.idx,token.idx+len(token.text),token.text)
                l_p_attributed = len(p_attributed)
                for material_pos in Material_poss:
                    if token.idx>=material_pos[0] and token.idx+len(token.text)<=material_pos[1]:
                        label = 'Material'
                        t = Token(token, label)
                        p_attributed.append(t)
                        # print(token.)
                        break
                if len(p_attributed) > l_p_attributed:
                    continue
                for value_pos in Value_poss:
                    if token.idx>=value_pos[0] and token.idx+len(token.text)<=value_pos[1]:
                        label = 'Value'
                        t = Token(token, label)
                        p_attributed.append(t)
                        break
                if len(p_attributed) > l_p_attributed:
                    continue
                label = 'None'
                t = Token(token, label)
                p_attributed.append(t)

            start = 0
            sentences = []
            for j in range(len(p_attributed)):
                if p_attributed[j].text == '.':
                    sentences.append(p_attributed[start:j-1])
                    start = j+1
            yield ins,sentences



if __name__ == "__main__":
    dir_path = "../data/dataset/test/"
    data = Dataset(dir_path)
    cnt = 0
    for ins,sentences in data.nerAnnoIter():
        for j in sentences:
            pass
            # print(j[0].pos)
            # for i in j:
            #     if i.label == 'Value':
            #         print(i.shape,'\t\t', i.text)
