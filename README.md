# HAABSA-Multidim-Multihead 

Source code for the master thesis: A Hybrid Approach for Aspect-Based Sentiment Analysis Using Multi-dimensional and Multi-head Attention

## How to run the code
- Install all dependencies and libraries, as explained in the base HAABSA (Wallaart and Frasincar (2019)) model. The description for this can be found at [https://github.com/ofwallaart/HAABSA](https://github.com/ofwallaart/HAABSA).
- Trusca et al. (2020) add BERT context dependent word embeddings. Procedure for producing these word embeddings can be found at https://github.com/mtrusca/HAABSA_PLUS_PLUS.
- Structure of the code is mostly the same. I have added:
	- **cross_validation_{model}.py** - this is the code that runs the cross validation for the models. The hyperparamaeters you want to use for grid search can be specified in this file.
	- **lcrModelAlt_{model}_cross.py** - This is code for running the cross validation versions of the model (such that it gives the validation error and validation accuracy). This file is called by **cross_validation_{model}.py**, so you do not have to use this file directly.
	-  **lcrModelAlt_{model}** - Code to run all the different models. These do not have to be called directly, they are called by the main method.
	- **main_consistency_test.py** - This method runs the training and testing procedure 100 times for the specified model in the file. It determines the average statistics (accuracy, st dev) between the 100 runs.
## Word embeddings
 - GloVe word embeddings (SemEval 2015): https://drive.google.com/file/d/14Gn-gkZDuTVSOFRPNqJeQABQxu-bZ5Tu/view?usp=sharing
 - GloVe word embeddings (SemEval 2016): https://drive.google.com/file/d/1UUUrlF_RuzQYIw_Jk_T40IyIs-fy7W92/view?usp=sharing
 - ELMo word embeddings (SemEval 2015): https://drive.google.com/file/d/1GfHKLmbiBEkATkeNmJq7CyXGo61aoY2l/view?usp=sharing
 - ELMo word embeddings (SemEval 2016): https://drive.google.com/file/d/1OT_1p55LNc4vxc0IZksSj2PmFraUIlRD/view?usp=sharing
 - BERT word embeddings (SemEval 2015): https://drive.google.com/file/d/1-P1LjDfwPhlt3UZhFIcdLQyEHFuorokx/view?usp=sharing
 - BERT word embeddings (SemEval 2016): https://drive.google.com/file/d/1eOc0pgbjGA-JVIx4jdA3m1xeYaf0xsx2/view?usp=sharing
 
Download pre-trained word emebddings: 
- GloVe: https://nlp.stanford.edu/projects/glove/
- Word2vec: https://code.google.com/archive/p/word2vec/
- FastText: https://fasttext.cc/docs/en/english-vectors.html

## References
- Trusca, M. M., Wassenberg, D., Frasincar, F., and Dekker, R. (2020).  A hybrid approachfor aspect-based sentiment analysis using deep contextual word embeddings and hier-archical attention.  InWeb Engineering - 20th International Conference,  ICWE 2020,Helsinki,  Finland,  June  9-12,  2020,  Proceedings,  volume  12128  ofLecture  Notes  inComputer Science, pages 365–380. Springer.
- Wallaart, O. and Frasincar, F. (2019).  A hybrid approach for aspect-based sentiment anal-ysis using a lexicalized domain ontology and attentional neural models.  In16th  Ex-tended  Semantic  Web  Conference  (ESWC  2019), volume 11503 ofLNCS, pages 363–378. Springer.
