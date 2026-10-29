# My write-up for the CS336 Assignment 4

## Problem (look_at_cc):  Looking at Common Crawl

(a) The URL is https://020bld.cn.vauofnj.cn/html/991f998999.html. I can't access it currently, on the date of October 25 2026. The page looks like it has some JavaScript but I can't understand what it is about since it's mostly in some Chinese-like language.

(b) Yes, "< Previous > Next" and "→" should be filtered out by the extractor in my opinion. Training a model like this could result in memorizing the HTML like structure and symbols that belong to HTML and we don't want in the response of a LM. The sentence and word structure and semantic meaning perhaps could be useful to the model to learn the language and grammar.

(c) The content I saw was I guess a news page from a chinese domain and it could be beneficial to the model for making it recent perhaps and not very beneficial when we want the model to learn sentence structure or acquire some worldly knowledge.

(d) It took about 15 examples for me to see a decent sentence that I could understand and that was in English. Most of the documents are non-educational content or non-english of which most are chinese and I also encountered into some russian and german.

## Problem (extract_text):  HTML to text conversion

(b) I've gone through 20 WARC files that have gone through the text extraction process and all of them were either Chinese, empty, or JSON-like things and none of them had any educational value. WET files, on the other hand, had more like a paragraph structure though still most of them were non-english therefore I wasn't able to understand their contents but I came across some content that has some information about a person in of the files.

## Problem (language_identification):  Language identification

(b) We could respond the user's question with a different language than the one he responded to us which could be a problem. We could perhaps run the language identifier to check the user's message and if our model is not trained on that language, we could just let the user know that and stop the generation rather than the model responding to something it doesn't understand.

(c) Based on my observations 0.6 could be a suitable confidence threshold since there were some examples of JSON objects that had other language inside the content but the variable names were english and model outputted 0.51 and also some of them were empty and model still outputted 0.11 but those shouldn't be a problem after thresholding.

## Problem (mask_pii):  Personally identifiable information

(d) The language model could learn to continue the sequence with the placeholder email/phone/IP instead of the user's own email/phone/IP since it always saw the emails/phones/IPs this way in its training set.

(e) One false negative I saw is "ip-10-67-67-113". I didn't see any false positives though.

## Problem (harmful_content):  Harmful content

(c) Perhaps, if we use these filters to create the training set, then the model never would see hateful speech or NSFW content and this could result in random outputs when a user enters these, instead we could synthetically introduce these toxic content to make the model learn that these are bad since how can it learn it's bad if it never sees these?

(d) I've look through 20 non-toxic or non-NSFW content and all of them were correctly labeled and also viewed one NSFW and toxic labeled example and believe it was correctly labeled based on the content even with a low threshold, this would probably work, since we don't want any NSFW content in general in our model in my opinion.

## Problem (gopher_quality_filters):  Gopher quality filters

(b) I don't disagree with the predictions where the text was quality based on the Gopher quality filter since not most of them passed but we need a way to still separate some parts of the quality text that passed it since it still has some artifacts from the web page in general even though the document has passed the test since we either pass the whole document or not. An ecommerce store that had somewhat useful information about their product may have been perhaps a little helpful, though I still agree with the whole text not passing the text.

## Problem (filter_data):  Filter data for language modeling

(a) In total we had 637,937 documents with 100 wet files I've downloaded (n_files = 100). ~2k (~0.3%) of them were toxic/nsfw, then ~169k of the remaining documents (~26%) did not pass the gopher quality filter, lastly ~224k of the remaining did not pass our quality filter. We kept ~242k of the ~637k documents which is ~38%.

(b) We used 100 raw files in our test and the script ran around 12 minutes. So, handout's 2500 files would take around 300 minutes and the full crawl which is around 100k wet files would take around 12000 minutes or 200 hours.
