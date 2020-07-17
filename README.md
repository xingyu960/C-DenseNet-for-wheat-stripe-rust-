# C-DenseNet-for-wheat-stripe-rust-
## Table of Contents
        1.Environment
        2.Download
        3.How2train
        4.Reference
## Environment
    ubuntu 16.04
    torch == 1.2.0
## Download
    We have provided the training results of C-DenseNet, you can use test.py to test the classification effect of the model：
    https://pan.baidu.com/s/1sqE1hwszvaozl06aKPd6wQ   Extraction code:tw36
    the test pictures are put in ./dataset/test/
## How2train
### Create an environment
    *conda create -n pytorch python=3.6
    *conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0
    *pip install -r requirements.txt
### train
    *Before training, you need to change all the file paths in train.py and dataset_nocsv.py to your path
    *The labels in dataset_nocsv.py need to be replaced with the label name of your training set
    *python train.py 
### test
    *Before testing, you need to change all the file paths in test.py and dataset_nocsv.py to your path
    *The class_names in test.py need to be consistent with the class name order during training
    *python test.py
## Reference
    https://blog.csdn.net/hacker_long/article/details/100138454?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522159495422819725211965258%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=159495422819725211965258&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_v1~rank_blog_v1-2-100138454.pc_v1_rank_blog_v1&utm_term=pytorch+%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB

    
