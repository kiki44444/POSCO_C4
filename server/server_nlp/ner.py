import nlp.dictionary as dictionary
import nlp.tune_data as tune_data
from ckonlpy.tag import Postprocessor
from ckonlpy.tag import Twitter


def ner_function(audio_sentence):
    twitter = Twitter()
    postprocessor = Postprocessor(twitter, ngrams = tune_data.ngrams)
    sen = postprocessor.pos(audio_sentence)

    nouns = []
    for l in sen:
        if l[1] == 'Noun':
            nouns.append(l[0])

    # 사전 불러오기
    artists = dictionary.artists
    artworks = dictionary.artworks
    dic = dictionary.dic

    List = [0, 0]
    for noun in nouns:
        if noun in artists:
            List[0] = noun
        elif noun in artworks:
            List[1] = noun

    if List[0] != 0 and List[1] != 0:
        # with open('text.txt', 'w') as f:
        #     f.write(dic[List[1]])
        #print('작품설명이 text.txt 파일로 저장되었습니다.')
        print('작품설명: ')
        return dic[List[1]]
    elif List[0] == 0 and List[1] != 0:
        # with open('text.txt', 'w') as f:
        #     f.write(dic[List[1]])
        #print('작품설명이 text.txt 파일로 저장되었습니다.')
        print('작품설명: ')
        return dic[List[1]]
    elif List[0] != 0 and List[1] == 0:
        # with open('text.txt', 'w') as f:
        #     f.write(dic[List[0]])
        # print('작가설명이 text.txt 파일로 저장되었습니다.')
        print('작가설명: ')
        return dic[List[0]]
    elif List[0] == 0 and List[1] == 0:
        return '작품을 인식 중 입니다. 조금만 기다려주세요.' # 이미지 디텍션
    else:
        raise Exception('CHECK THIS # 2')