from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse

#additional library required
import tensorflow_hub as hub
import numpy as np
import nltk
from nltk import tokenize
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity
import json
from collections import defaultdict

# Create your views here.

# nltk.download('punkt')

#loading universal_text_encoder_model
# module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
# embed = hub.load("/var/www/text_searcher/use model")
embed = hub.load("/var/www/coeapps/use_model")

def home(request):
  if request.method == 'GET':
    return render(request,'homepage/index.html')
  else :
    return render(request,'homepage/index.html')


#function for scanning a corpus..
def scan(request):
  # print("running")
  if request.method == 'POST':
      ## access you data by playing around with the request.POST object
      corpus = request.POST["corpus"]
      text2scan = request.POST["text2scan"]
      namechecked = int(request.POST["namechecked"])

      strlen = len(text2scan.split())
      text2scan_raw = text2scan
      text2scan = [text2scan]
      print(text2scan)
      print(strlen)

      text2scan = embed(text2scan)
          
      if (strlen == 0) :
          return JsonResponse({'matched': "none",'score': "none" })

      elif(namechecked == 1) :
          print("running namechecked")
          matched = getmatches(text2scan_raw, corpus)
          score = 0
          print(matched)
          return JsonResponse({'matched': matched,'score': score })

      elif(strlen == 1) :
          words = tokenize.word_tokenize(corpus)
          words_embeddings = embed(words)
          print("running semantic search")
          res = cosine_similarity(text2scan,words_embeddings)
          res = res[0]
          maxx = res.argmax()

          matched = words[maxx]
          score = res[maxx]*100
          score = json.dumps(score.astype(float)) 
          print(matched+"  "+score)
          return JsonResponse({'matched': matched,'score': score })

      # for sentences...
      else :
          sentences = tokenize.sent_tokenize(corpus)
          sentences_embeddings = embed(sentences)
          print("running sentences")
          res = cosine_similarity(text2scan,sentences_embeddings)

          res_flat=res[0]
          val=max(res_flat)/2
          resset = []
          for i, score in enumerate(res_flat):
             if score > val:
                resset.append([i,score])

          resset = sorted(resset,key=lambda x:x[1],reverse=True)

          for n,i in enumerate(resset):
            sen = sentences[i[0]]
            if len(tokenize.word_tokenize(sen))> 10:
                mergers = [" and "," as "," but "," or "," for "," yet "," therefore "," moreover "," thus ", "," , ";" ]
                parts = [sen]

                for key in mergers:
                  part = sen.split(key)
                  for j in part:
                    if len(tokenize.word_tokenize(j)) > 3:
                      if j not in parts:
                        parts.append(j)
                # print(parts)
                emdings = embed(parts)
                result = cosine_similarity(text2scan,emdings)
                maxx = max(result[0])
                if maxx > i[1]:
                  resset[n][1] = maxx
                  # print(result, i[0])

          resset = sorted(resset,key=lambda x:x[1],reverse=True)
          resset
  
          matched = sentences[resset[0][0]]
          score = resset[0][1]*100
          score = json.dumps(score.astype(float)) 
          print(matched+"  "+score)
          return JsonResponse({'matched': matched,'score': score })
  
  else :
     return redirect(home)

      

# for uploading files
def upload_file(request):

    doc = Document(request.FILES['myfile'])
    file_corpus = []
    for para in doc.paragraphs:
        file_corpus.append(para.text)
    file_corpus = '\n'.join(file_corpus)
    return render(request,'homepage/index.html',{'file_corpus': file_corpus})




# function for searching names..
def get_soundex(name):
    name = name.upper()

    soundex = ""
    soundex += name[0]

    ref_dict = {"BFPV": "1", "CGJKQSXZ": "2", "DT": "3",
                "L": "4", "MN": "5", "R": "6", "AEIOUHWY": "."}

    dictionary = {i: v for x, v in ref_dict.items() for i in x}

    for char in name[1:]:
        if dictionary.get(char) != None:
            code = dictionary[char]
            if code != soundex[-1]:
                soundex += code

    soundex = soundex.replace(".", "")
    soundex = soundex[:4].ljust(4, "0")

    return soundex


def getmatches(queryStr, allNames):
    dict_processed = defaultdict(lambda: [])
    list_of_names = allNames.split()
    queryCode = get_soundex(queryStr)
    res = map(get_soundex, list_of_names)

    for code, name in zip(res, list_of_names):
        dict_processed[code].append(name)

    # print(*dict_processed.items(),sep="\n")

    return dict_processed.get(queryCode)

