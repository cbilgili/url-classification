# url-classification
This project can classify URLs to "index type" or "detail type" ,and it just depend on the features that are extracted by pure URL .


the predict successful rate ,index type url:91% , detail type url：94%



how to use it :

1. make train set 
    info_url_file : This file is the "detail type " set of urls .
    index_url_file : This file is the "index type " set of urls .

2. make test set 
    test_url_file  : tail index_url_file -n1000

3. execute the python shell

    i.python feature_engine.py train  ,to train the svm model 

    ii. python feature_engine.py  test_file 0 test_url_file |grep "1.0 \[" 
        python feature_engine.py  test_file 1 test_url_file_info |grep "0.0 \["

            to predict the effection of the svm model. and show the wrong classifations 
