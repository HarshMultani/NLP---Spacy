import spacy

nlp = spacy.load('en_core_web_md')



## Advanced NLP with spacy
## Chapter 1 started ----------------------------------------------------------------------------------------------
## Code for analysing lexical attributes and linguistic annotations Started ---------------------------------------


#doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
#print(type(doc))


## Code for testing Lemmatization, Syntactic Dependency, Stopwords, Part of Speech tagger
#for token in doc:
    #print(token.text, token.lemma_,  token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop
    

## Code for testing Named Entity Recognition
#for ent in doc.ents:
    #print(ent.text, ent.start_char, ent.end_char, ent.label_)


## Word Vectors in spacy
#tokens = nlp("dog cat banana afskfsd")
#for token in tokens:
    #print(token.text, token.has_vector, token.vector_norm, token.is_oov)


## Sentence Separation in spacy
#doc = nlp("Peach emoji is where it has always been. Peach is the superior "
          #"emoji. It's outranking eggplant 🍑 ")
#sentences = list(doc.sents)
#assert len(sentences) == 3
#print(sentences[1].text)


## 
## Example
#doc = nlp("Hello World!")

#print(doc)
#for token in doc:
    #print(token.i)
    #print(token.text)

## 
## Code for analysing lexical attributes and linguistic annotations Ended -----------------------------------------------------------


## Code for Rule based matching started ---------------------------------------------------------------------------------------------
## Rule based matching

"""
from spacy.matcher import Matcher

matcher = Matcher(nlp.vocab)
pattern = [{"TEXT":"iPhone"},{"TEXT":"X"}]
matcher.add("IPHONE_PATTERN", None, pattern)
doc = nlp("Upcoming iPhone X release date leaked")
matches = matcher(doc)

for match_id, start, end in matches:
    # Get the matched span
    matched_span = doc[start:end]
    print(matched_span.text)


pattern = [
    {"IS_DIGIT": True},
    {"LOWER": "fifa"},
    {"LOWER": "world"},
    {"LOWER": "cup"},
    {"IS_PUNCT": True}
]


doc = nlp("2018 FIFA World Cup: France won!")
#matches = matcher(doc)
matcher.add("EXAMPLE_PATTERN", None, pattern)
print(matches)

for match_id, start, end in matches:
    # Get the matched span
    matched_span = doc[start:end]
    print("Here")
    print(matched_span.text)


pattern = [
    {"LEMMA": "love", "POS": "VERB"},
    {"POS": "NOUN"}
]
doc = nlp("I loved dogs but now I love cats more.")
matcher.add("LOVE_NOUN", None, pattern)

for match_id, start, end in matches:
    # Get the matched span
    matched_span = doc[start:end]
    print("Here")
    print(matched_span.text)


pattern = [
    {"LEMMA": "buy"},
    {"POS": "DET", "OP": "?"},  # optional: match 0 or 1 times
    {"POS": "NOUN"}
]
doc = nlp("I bought a smartphone. Now I'm buying apps.")
matcher.add("BUY_NOUN", None, pattern)

for match_id, start, end in matches:
    # Get the matched span
    matched_span = doc[start:end]
    print("Here")
    print(matched_span.text)

"""

## Code for Rule based matching Ended ------------------------------------------------------------------------------------------
## Chapter 1 Ended -------------------------------------------------------------------------------------------------------------


## Chapter 2 Started ------------------------------------------------------------------------------------------------------------

## Vocabulary and String Store
## Strings and hashes
#coffee_hash = nlp.vocab.strings["coffee"]
#print(coffee_hash)
#string = nlp.vocab.strings[3197928453018144401]
#print(string)


## Lexeme object
#doc = nlp("I love coffee")
#lexeme = nlp.vocab["coffee"]
#print(lexeme.text, lexeme.orth, lexeme.is_alpha)


## Word Vectors and Similiarity
#doc1 = nlp("I like fast food")
#doc2 = nlp("I like pizza")
#print(doc1.similarity(doc2))


#doc = nlp("I have a banana")
#print(doc[3].vector)

## Chapter 2 Ended --------------------------------------------------------------------------------------------------------------------


## chapter 3 Started ------------------------------------------------------------------------------------------------------------------

## Custom Pipeline components

#def custom_component(doc):
    #print("Doc Length:", len(doc))
    #return doc

#nlp.add_pipe(custom_component, first=True)

#print("Pipeline:", nlp.pipe_names)
#doc = nlp("Hello world!")
#print(doc)

## Chapter 3 Ended ---------------------------------------------------------------------------------------------------------------------