#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import json
import urllib
import re
import time
import random
import urlparse 
from svmutil import *

reload(sys)
sys.setdefaultencoding('utf-8')

#label_0 is index web page
train_label_0_num = 100000
#label_1 is infomation web page
train_label_1_num = 100000

predict_label_0_num = 20000 
predict_label_1_num = 20000 

IS_NORMALIZTION = True 
IS_DEBUG = True

#svm_parameter_str = '-c 4 -s 4 -t 0'
svm_parameter_str = '-c 4 -g 0.100077193 -h 0 '

#index_url_file = 'index_url_file'
#info_url_file = 'info_url_file_2'
index_url_file = 'data/all_subdomain_index_random_20w'
info_url_file = 'data/all_subdomain_info_random_20w'
test_index_url_file = 'data/all_subdomain_index_random_20w'
test_info_url_file = 'data/all_subdomain_info_random_20w'

global feature_index_map 
feature_index_map = {} # feature key => key index ,index start from 0

global dir_name_index_map 
dir_name_index_map = {} # feature key => key index ,index start from 0

global doc_name_index_map 
doc_name_index_map = {} # feature key => key index ,index start from 0

global ext_name_index_map 
ext_name_index_map = {} # feature key => key index ,index start from 0

global params_name_index_map 
params_name_index_map = {} # feature key => key index ,index start from 0

global union_name_score_map
union_name_score_map = {} # feature key => score  ,index id 9 

global dir_name_score_map
dir_name_score_map = {} # feature key => score  ,index id 11

global res_name_score_map
res_name_score_map = {} # feature key => score  ,index id 12

global para_name_score_map
para_name_score_map = {} # feature key => score  ,index id 13

global para_value_score_map
para_value_score_map = {} # feature key => score  ,index id 14

global resname_digitlen_score_map
resname_digitlen_score_map = {} # feature key => score  ,index id 17

global paravalue_digitlen_score_map
paravalue_digitlen_score_map = {} # feature key => score  ,index id 17

global tmp_result_map
tmp_result_map = {} # feature key => score  ,index id 17



infoid_params_name_list = ['aid','newsid','id','docid','itemid','tid']
indexid_params_name_list\
=['pageid','catid','cateid','typeid','sortid','classid','fid','c_id','class_id','columnid','Channelid','subid']

def choose_feature_fuc( fuc_type = 1):
    if fuc_type == 0:
        return feature_select_first
    return feature_select_by_soso 

def make_svm_feature_test_file(input_url_file , out_feature_file ,url_type ,label_num):
    feature_select_fuc = choose_feature_fuc()
    of = open(out_feature_file,'w') 
    f_l = open(input_url_file,'r').readlines()
    f_l = random.sample(f_l,label_num)
    doc_num = 0
    for l in f_l :
        if doc_num >= label_num:
            break
        url = l.strip()
        doc_num += 1
        output_line = feature_select_fuc(str(url_type) , url)
        of.write(output_line)
    of.close()

def make_svm_feature_file(input_index_file ,input_info_file, out_file ,label_0_num ,label_1_num):
    feature_select_fuc = choose_feature_fuc()
    debug_f = open('debug_log','w') 
    of = open(out_file,'w') 
    f_l = open(input_index_file,'r').readlines()
    f_l = random.sample(f_l,label_0_num)
    doc_num = 0
    for l in f_l :
        if doc_num >= label_0_num:
            break
        url = l.strip()
        doc_num += 1
        output_line = feature_select_fuc('0' , url)
        of.write(output_line)
        debug_f.write(url +'\t' +output_line)
    f_l = open(input_info_file,'r').readlines()
    f_l = random.sample(f_l,label_1_num)
    doc_num = 0
    for l in f_l :
        if doc_num >= label_1_num:
            break
        url = l.strip()
        doc_num += 1
        output_line = feature_select_fuc('1' , url)
        of.write(output_line)
        debug_f.write(url +'\t' +output_line)
    of.close()
    debug_f.close()
    
def train_paramters():    
    label_0_num = 5000
    label_1_num = 10000
    make_svm_feature_file(index_url_file ,info_url_file, 'feature_file' ,label_0_num,label_1_num)
    y, x = svm_read_problem('feature_file')
    train_set_num = 2000
    m = svm_train(y[label_0_num -train_set_num:label_0_num+train_set_num*2], x[label_0_num\
    -train_set_num:label_0_num+train_set_num*2], '-h 0 -v 5 -t 2 -g 2')
    #-train_set_num:label_0_num+train_set_num*2], '-c 4 -s 0 -v 20')
    return m

def train_predict_random(feature_file,label_0_num ,label_1_num):    
    y, x = svm_read_problem(feature_file)
    print 'svm_read_problem ' ,feature_file ,'finish'
    m = svm_train(y, x,svm_parameter_str )
    print 'svm_train ' ,feature_file ,'finish'
    make_svm_feature_test_file(index_url_file , 'index_feature_file' ,0,label_0_num)
    print 'make_svm_feature_test_file ' , index_url_file , 'random' ,label_0_num
    y, x = svm_read_problem('index_feature_file')
    p_label, p_acc, p_val = svm_predict(y, x, m)
    make_svm_feature_test_file(info_url_file , 'info_feature_file' , 1 ,label_1_num)
    print 'make_svm_feature_test_file ' , info_url_file , 'random' ,label_1_num
    y, x = svm_read_problem('info_feature_file')
    p_label, p_acc, p_val = svm_predict(y, x, m)
    return m
    
def train_predict(feature_file,label_0_num ,label_1_num):    
    y, x = svm_read_problem(feature_file)
    train_set_num = 2000
    m = svm_train(y[label_0_num -train_set_num:label_0_num+train_set_num*2], x[label_0_num\
    -train_set_num:label_0_num+train_set_num*2], svm_parameter_str)
    p_label, p_acc, p_val = svm_predict(y[:train_set_num], x[:train_set_num], m)
    p_label, p_acc, p_val = svm_predict(y[label_0_num+train_set_num*2:label_0_num+train_set_num*3],    x[label_0_num+train_set_num*2:label_0_num+train_set_num*3], m)
    return m

def init():
    #load config

    global union_name_score_map,dir_name_score_map
    global res_name_score_map, para_name_score_map, para_value_score_map

    union_name_score_map = make_key_rate('conf/feature_url.txt',line_key = '9_')
    dir_name_score_map = make_key_rate('conf/feature_url.txt',line_key = '11_')
    res_name_score_map = make_key_rate('conf/feature_url.txt',line_key = '12_')
    para_name_score_map = make_key_rate('conf/feature_url.txt',line_key = '13_')
    para_value_score_map = make_key_rate('conf/feature_url.txt',line_key = '14_')


def max_continuity_digit(test_str):
    max_continuity_digit = 0
    num_len = 0
    for c in test_str:
        if c.isdigit():
            num_len += 1
        else :
            max_continuity_digit = max_continuity_digit if max_continuity_digit > num_len else num_len
            num_len = 0
    max_continuity_digit = max_continuity_digit if max_continuity_digit > num_len else num_len
    return max_continuity_digit 

#TODO 
def is_contain_date_string(path):
    dir_names = path.split('/')
    is_date_string = 0
    #"%Y%m%d","%Y-%m-%d",%Y/%m/%d,%Y%m/%d,2015-12/30/,2015/1230,/2015/12-30/
    #/detail_2015_12/29/,
    for dir_name in dir_names :
        if not dir_name.startswith('20'):
            continue
        tmp_date_str = dir_name[:6]
        try: 
            time.strptime(tmp_date_str,'%Y%m')
            is_date_string = 1
            break
        except Exception,e: 
            pass
        tmp_date_str = dir_name[:7]
        try: 
            time.strptime(tmp_date_str,'%Y-%m')
            is_date_string = 1
            break
        except Exception,e: 
            pass

    return is_date_string

def is_exist_doc_name(dir_names):
    value = 0
    n = len(dir_names)
    if n > 0 and dir_names[n-1] == '':
        value = -1
        if n > 1 and dir_names[n-2].isdigit():
            value = 1 
    '''
    if n > 0 and dir_names[n-1] == '':
        value = -1
    else:
        value = 1'''
    return value

def calc_paramter_name( query):
    return 0

    # just load different type score from conf/feature_url.txt
def make_key_rate(conf_file,line_key ):
    f = open(conf_file,'r')
    conf_start = False
    conf_type_name = '' 
    key_rate_map = {}
    for l in f :
        l = l.strip()
        if l.startswith('['):
            conf_start = True
            conf_type_name = l[1:len(l)-1]
            continue
        if not conf_type_name.startswith(line_key) :
            continue
        arr_str = l.split()
        if len(arr_str)!= 3 :
            continue
        key_rate_map[arr_str[2]] = (float("%.6f"  % (float(arr_str[0]) )),float("%.6f"  %  (float(arr_str[1]) )))
        total_score =  float(arr_str[0]) +   float(arr_str[1])
        #key_rate_map[arr_str[2]] = (float("%.6f"  % (float(arr_str[0]) /total_score)) ,float("%.6f"  % (float(arr_str[1]) /total_score)))
    return key_rate_map

    
def path_name_keyword(dir_names):
    index_prefix_key_list = ['index','category','list','search']
    info_prefix_key_list = ['detail','content','system','news','article']
    info_key_list = ['a']
#TODO add moew keyword
    file_name =  dir_names[len(dir_names)-1] 
    value = 0 
    for dir_name in dir_names:
        if dir_name in info_key_list :
            value = 1
            break
        for tmp_key in index_prefix_key_list :
            if dir_name.startswith(tmp_key):
                value = -1 
                break
        if value != 0 :
            break
        for tmp_key in info_prefix_key_list :
            if dir_name.startswith(tmp_key):
                value = 1 
                break
        if value != 0 :
            break
    return value 

    
def classify_doc_type(path):
    #info_key_list = ['id','','shtml','shtm','jhtml']
    #index_key_list = ['fid','catid','shtml','shtm','jhtml']
    dir_names = path.split('/')

    index_key_list = ['php','aspx','jsp','do']
    info_key_list = ['html','htm','shtml','shtm','jhtml']
    ext_name =  dir_names[len(dir_names)-1] 
    if '.' in ext_name:
        ext_name = ext_name[ext_name.rfind('.')+1:]
    else:
        ext_name = ''
    result_type = 0
    if ext_name != '':
        if ext_name in info_key_list :
            result_type = 1
        elif ext_name in index_key_list:
            result_type = -1
    return result_type

    # 0 . paramters nums 
def calc_paramter_number(query_str):
    if query_str == '':
        return 0
    query_params = query_str.split('&')
    max_paramter_num = 6
    if IS_NORMALIZTION :
        return "%.2f" % (float(len(query_params))  / max_paramter_num)
    else:
        return len(query_params)
    
    # 2 . all dir names chars length 
def calc_dirnames_length(dir_names):
    value = 0 
    if  len(dir_names)<3:
        return value 
    for i in range (1, len(dir_names) - 1):
        value += len(dir_names[i])
    if IS_NORMALIZTION :
        return "%.2f" % ( float( value) / 30)
    else:
        return value

    # 3 . average char length ???? 
def average_dirnames_length(dir_names):
    value = 0 
    if  len(dir_names)<3:
        return value 
    for i in range (1, len(dir_names) - 1):
        value += len(dir_names[i])

    if IS_NORMALIZTION :
        return "%.2f" % ( float( value)/(len(dir_names)-2 )/ 10)
    else:
        return "%.2f" % ( float( value)/(len(dir_names)-2 ))

    # 4. doc type judeyment? static or dynamic pages
def is_dynamic_page(query):
    if query is not None and query != '':
        return 1
    return 0

    # 5. whether the page is default page??
def is_default_page(dir_names):
    dafault_keyword = ['index','default']
    if len(dir_names) > 0 :
        file_name =  dir_names[len(dir_names)-1]
        if '.' in file_name:
            file_name = file_name[:file_name.rfind('.')]
        if file_name in dafault_keyword:
            return 1
    return 0

    
    # 6. the resource file name is digit ?
def is_digit_filename(dir_names,query):
    file_name = ''
    if len(dir_names) > 0 :
        file_name =  dir_names[len(dir_names)-1]
        if '.' in file_name:
            file_name = file_name[:file_name.rfind('.')]
    num = len(file_name)
    value = 0
    if num > 0 and file_name.isdigit() :
        value = 0.5
        if query is not None and query != '':
            value += 0.1
        if num > 4 :
            value += 0.4
        elif num > 3:
            value += 0.3
        elif num <= 3:
            value += 0.1
    return value 

    # 7  is contain date format for resource name 
def is_match_date_filename( file_name):
    value = 0
    #"%Y%m%d","%Y-%m-%d",%Y/%m/%d,%Y%m/%d,2015-12/30/,2015/1230,/2015/12-30/
    if '.' in file_name:
        file_name = file_name[:file_name.rfind('.')]
    if len(file_name) < 6 :
        return 0
    max_continuity_digit = 0
    num_len = 0
    for c in file_name:
        if c.isdigit():
            num_len += 1
        else :
            max_continuity_digit = max_continuity_digit if max_continuity_digit > num_len else num_len
            num_len = 0
    max_continuity_digit = max_continuity_digit if max_continuity_digit > num_len else num_len
    if max_continuity_digit  < 2 :
        return 0
    
    for i in range (0,len(file_name)-7):
        if file_name[i] == '2' and file_name[i+1] == '0' and \
            (file_name[i+2] == '0' or file_name[i+2] == '1') :
            if (i + 8) > len(file_name):
                break
            tmp_date_str = file_name[i:i+8]
            try: 
                time.strptime(tmp_date_str,'%Y%m%d')
                value = 1
                break
            except Exception,e: 
                pass
            tmp_date_str = file_name[i:10]
            try: 
                time.strptime(tmp_date_str,'%Y-%m-%d')
                value = 1
                break
            except Exception,e: 
                pass
            tmp_date_str = file_name[i:10]
            try: 
                time.strptime(tmp_date_str,'%Y_%m_%d')
                value = 1
                break
            except Exception,e: 
                pass
    return value
            
    # 8  is contain date format for dir name 
def is_match_date_dirnames( dir_names):
    dir_str = ''.join(dir_names)
    value = 0
    #"%Y%m%d","%Y-%m-%d",%Y/%m/%d,%Y%m/%d,2015-12/30/,2015/1230,/2015/12-30/
    if len(dir_str) < 6 :
        return 0
    for i in range (0,len(dir_str)-5):
        if dir_str[i] == '2' and dir_str[i+1] == '0' and \
            (dir_str[i+2] == '0' or dir_str[i+2] == '1') :
            if (i + 5) > len(dir_str):
                break
            tmp_date_str = dir_str[i:i+6]
            try: 
                time.strptime(tmp_date_str,'%Y%m')
                value = 1
                break
            except Exception,e: 
                pass
            tmp_date_str = dir_str[i:i+7]
            try: 
                time.strptime(tmp_date_str,'%Y-%m')
                value = 1
                break
            except Exception,e: 
                pass
            tmp_date_str = dir_str[i:i+7]
            try: 
                time.strptime(tmp_date_str,'%Y_%m')
                value = 1
                break
            except Exception,e: 
                pass
    return value

    # 9. Union? 
def is_match_union(path,query ):
    global union_name_score_map
    if query != "" :
        path = path + "?" + query
    for key,value in union_name_score_map.items():
        static_arr = re.compile("\.\*|\*|\$").split(key)
        variable_arr = []
        i = 0
        while i < len(key) :
            if key[i] == "*" or key[i] == "$":
                variable_arr.append(key[i])
            i += 1
        tmp_variable_list = []
        pos = 0
        i = 0
        stop_flag = False
        while pos < len(path) and i < len(static_arr):
            if static_arr[i] not in path[pos:]:
                stop_flag = True
                tmp_variable_list = []
                break
            pos_2 = path[pos:].find(static_arr[i]) + pos 
            tmp_variable_list.append(path[pos:pos_2])
            pos = pos_2 + len(static_arr[i]) 
            i += 1
        if stop_flag == True or len(tmp_variable_list) == 0 or len(tmp_variable_list) !=\
            (len(variable_arr) + 1):
            continue
        i = 0
        match_num = 0
        for i in range (0 , len(variable_arr)) :
            if variable_arr[i] == "$" and tmp_variable_list[i+1].isdigit():
                match_num += 1
            elif len(tmp_variable_list[i+1]) > 0 :
                match_num += 1
        if match_num == len(variable_arr) :
            #print path,key,tmp_variable_list,variable_arr
            if value[0] > 0.8 :
                return -1
            elif value[1] > 0.8:
                return 1
            return 0
    return 0
    

def tool_plus_score(score_map,key,ori_score,load_tuple_index):
    ori_score += score_map[key][load_tuple_index]
    return ori_score
    
    # 11. match keyword in dri names 
def match_keyword_dirnames(dir_names,load_tuple_index = 0):
    global dir_name_score_map
    tmp_score = float(0)
    match_num = 0
    for dir_name in dir_names :
        if dir_name in dir_name_score_map.keys():
            tmp_score = tool_plus_score(dir_name_score_map,dir_name,tmp_score,load_tuple_index)
            match_num += 1
    if match_num == 0 :
        return 0
    return tmp_score / match_num

    # 12. match keyword in resource names 
def match_keyword_filenames(res_name,load_tuple_index = 0):
    global res_name_score_map
    tmp_score = float(0)
    match_num = 0
    res_arr = re.compile("\.|_|-").split(res_name)
    for name in  res_arr:
        if name in res_name_score_map.keys():
            tmp_score = tool_plus_score(res_name_score_map,name,tmp_score,load_tuple_index)
            match_num += 1
    if match_num == 0 :
        return 0
    return tmp_score / match_num

    # 13. match keyword in paramter names 
def match_keyword_paramters(query_str,load_tuple_index = 0):
    global para_name_score_map
    tmp_score = float(0)
    match_num = 0
    query_params = query_str.split('&')
    for para_k_v in  query_params:
        if '=' not in para_k_v :
            continue
        name = para_k_v[:para_k_v.find('=')]
        if name in para_name_score_map.keys():
            tmp_score = tool_plus_score(para_name_score_map,name,tmp_score,load_tuple_index)
            match_num += 1
    if match_num == 0 :
        return 0
    return tmp_score / match_num


    # 14. match paramter value
def match_keyword_para_value(query_str,load_tuple_index = 0):
    global para_value_score_map
    tmp_score = float(0)
    match_num = 0
    query_params = query_str.split('&')
    for para_k_v in  query_params:
        if '=' not in para_k_v :
            continue
        value = para_k_v[para_k_v.find('=')+1:]
        if value in para_value_score_map.keys():
            tmp_score = tool_plus_score(para_value_score_map,value,tmp_score,load_tuple_index)
            match_num += 1
    if match_num == 0 :
        return 0
    return tmp_score / match_num

    
    # 16. the length when the resource file name is digit,
def is_digit_filename_len(dir_names):
    file_name = ''
    if len(dir_names) > 0 :
        file_name =  dir_names[len(dir_names)-1]
        if '.' in file_name:
            file_name = file_name[:file_name.rfind('.')]
    num = len(file_name)
    value = 0
    if num > 0 and file_name.isdigit() :
        if num > 4 :
            value = 1 
        else:
            value = 0.5
    return value 
    
    # 18. the length of file name or last dir name ,which is the most long serial number 
def serial_number_length_infilename(dir_names):
    file_name = ''
    value = 0
    if len(dir_names) <= 1 :
        return value
    file_name =  dir_names[len(dir_names)-1]
    is_dir_name = False
    if file_name == '' : 
        # last dir name
        is_dir_name = True
    if is_dir_name :
        file_name =  dir_names[len(dir_names)-2] 
    else:
        if '.' in file_name:
            file_name = file_name[:file_name.rfind('.')]

    if file_name == '' :
        return value

    num = max_continuity_digit( file_name)
    if num >= 8 :
        value = 1 
    elif num >= 6 :
        value = 0.9
    elif num > 4 :
        value = 0.8
    elif num == 4 :
        value = 0.4
    return value 

def para_value_number_len(query_str):
    if query_str == '':
        return 0
    query_params = query_str.split('&')
    for query in query_params :
        pos = query.find('=')
        if pos <= 0 :
            continue
        key = query[:pos]
        p_value = query[pos+1:]
        
        if key not in infoid_params_name_list :
            continue
        num = len(p_value)
        value = 0
        if num > 0 and p_value.isdigit() :
            if num >= 8 :
                value = 1 
            elif num >= 6 :
                value = 0.9
            elif num > 4 :
                value = 0.8
            elif num == 4 :
                value = 0.4
        return value 
    return 0


def feature_select_by_soso(label_name , url):
    # output is list of eg:+1 1:0.708333 2:1 
    # the feature
    #scheme :// host / path / document . extension ? query=fragment
    index_value_list = []
    url = url.lower()
    url_portions = urlparse.urlparse(url)
    dir_names = url_portions.path.split('/')
    # 0 . paramters nums 
    tmp_index = 0 
    tmp_value = calc_paramter_number(url_portions.query)
    #index_value_list.append((tmp_index,tmp_value))
    # 1. path deepth 
    tmp_index = 1 
    tmp_value = len(dir_names) -1 
    if IS_NORMALIZTION :
        tmp_value = "%.2f" % ( float(len(dir_names) - 1 )/6)
    index_value_list.append((tmp_index,tmp_value))
    # 2 . all dir names chars length 
    tmp_index = 2 
    tmp_value = calc_dirnames_length(dir_names)
    index_value_list.append((tmp_index,tmp_value))
    # 3 . average char length ???? 
    tmp_index = 3 
    tmp_value = average_dirnames_length(dir_names)
    index_value_list.append((tmp_index,tmp_value))
    # 4. doc type judeyment? static or dynamic pages
    tmp_index = 4 
    tmp_value = is_dynamic_page(url_portions.query)
    index_value_list.append((tmp_index,tmp_value))
    # 5. whether the page is default page??
    tmp_index = 5 # is_default_page -> classify_doc_type
    tmp_value = classify_doc_type(url_portions.path)
    index_value_list.append((tmp_index,tmp_value))
    # 6. the resource file name is digit ?
    tmp_index = 6
    tmp_value = is_digit_filename(dir_names,url_portions.query)
    index_value_list.append((tmp_index,tmp_value))
    # 7. is contain date format in resource file name  ?
    tmp_index = 7
    tmp_value = is_match_date_filename(dir_names[len(dir_names)-1])
    index_value_list.append((tmp_index,tmp_value))
    # 8. is contain date format in dir names  ?
    tmp_index = 8 
    tmp_value = is_match_date_dirnames(dir_names[:len(dir_names)-1])
    index_value_list.append((tmp_index,tmp_value))
    # 9. Union? 
    tmp_index = 9 # TODO
    tmp_value = 0 # is_match_union(url_portions.path , url_portions.query) 
    index_value_list.append((tmp_index,tmp_value))
    # 10 pattern match 
    tmp_index = 10 # TODO
    tmp_value = 0 #match_pattern_result() 
    index_value_list.append((tmp_index,tmp_value))
    '''# 11. match keyword in dri names 
    tmp_index = 11 
    tmp_value = match_keyword_dirnames(dir_names[:len(dir_names)-1],0)
    index_value_list.append((tmp_index,tmp_value))
    # 12. match keyword in resource names 
    tmp_index = 12
    tmp_value = match_keyword_filenames(dir_names[len(dir_names)-1],0)
    index_value_list.append((tmp_index,tmp_value))
    # 13. match keyword in paramter names 
    tmp_index = 13
    tmp_value = match_keyword_paramters(url_portions.query,0)
    index_value_list.append((tmp_index,tmp_value))
    # 14. match paramter value
    tmp_index = 14
    tmp_value = match_keyword_para_value(url_portions.query,0)
    index_value_list.append((tmp_index,tmp_value))'''
    # 15. last dir name ,length 
    tmp_index = 15
    tmp_value = len(dir_names[len(dir_names)-2]) if len(dir_names)>= 2 else 0
    if IS_NORMALIZTION :
        tmp_value = float(tmp_value) / 10
    index_value_list.append((tmp_index,tmp_value))
    # 17. url match "list" 
    tmp_index = 16 #TODO 
    tmp_value = -1 if 'list' in url_portions.path else 0
    #index_value_list.append((tmp_index,tmp_value))
    # 18. the length of file name or last dir name ,which is the most long serial number 
    tmp_index = 17#
    tmp_value = serial_number_length_infilename(dir_names)
    index_value_list.append((tmp_index,tmp_value))
    # 19. the length of paramter value ,docid ,cid ,aid,
    tmp_index = 18#
    tmp_value = para_value_number_len(url_portions.query)
    index_value_list.append((tmp_index,tmp_value))
    # 111. match keyword in dri names 
    tmp_index = 111 
    tmp_value = match_keyword_dirnames(dir_names[:len(dir_names)-1],1)
    index_value_list.append((tmp_index,tmp_value))
    # 112. match keyword in resource names 
    tmp_index = 112
    tmp_value = match_keyword_filenames(dir_names[len(dir_names)-1],1)
    index_value_list.append((tmp_index,tmp_value))
    # 113. match keyword in paramter names 
    tmp_index = 113
    tmp_value = match_keyword_paramters(url_portions.query,1)
    index_value_list.append((tmp_index,tmp_value))
    # 114. match paramter value
    tmp_index = 114
    tmp_value = match_keyword_para_value(url_portions.query,1)
    index_value_list.append((tmp_index,tmp_value))
    #print index_value_list
    output_strings = []
    for index_value in index_value_list:
        output_strings.append(str(index_value[0]) + ':' + str(index_value[1]))
    return label_name + ' ' + ' '.join(output_strings) +'\n'

            
def feature_select_first(label_name , url):
    # output is list of eg:+1 1:0.708333 2:1 
    #url split by / 
    #catch key str and dir index
    #scheme :// host / path / document . extension ? query=fragment
    index_value_list = []
    url_portions = urlparse.urlparse(url)
    dir_names = url_portions.path.split('/')
    # 0 . url path length 
    tmp_index = 0 
    path_len = url_portions.path.startswith('/') and len(url_portions.path) -1 or len(url_portions.path)
    tmp_value = "%.2f" % ( float(path_len)/50)
    index_value_list.append((tmp_index,tmp_value))
    # 1. is contain date string ?
    tmp_index = 1
    tmp_value = is_contain_date_string(url_portions.path)
    index_value_list.append((tmp_index,tmp_value))
    # 2. is exist document name ?
    tmp_index = 2
    tmp_value = is_exist_doc_name(dir_names)
    index_value_list.append((tmp_index,tmp_value))
    # 3. doc type judeyment  ?
    tmp_index = 3
    tmp_value = classify_doc_type(url_portions.path)
    index_value_list.append((tmp_index,tmp_value))
    # 4. is exist  keyword paramter names  ?
    tmp_index = 4 
    tmp_value = calc_paramter_name(url_portions.query)
    index_value_list.append((tmp_index,tmp_value))
    '''# 5. paramter number 
    tmp_index = 5 
    tmp_value = calc_paramter_number(url_portions.query)
    index_value_list.append((tmp_index,tmp_value))'''
    # 6. path deepth 
    tmp_index = 6 
    tmp_value = "%.2f" % ( float(len(dir_names) - 1 )/6)
    index_value_list.append((tmp_index,tmp_value))

    # 7. parmter value : list or content 
    '''tmp_index = 6 
    tmp_value = (url_portions.path)
    index_value_list.append((tmp_index,tmp_value))'''
    # 8. file or dir name: index url , eg: /index/,// 
    tmp_index = 7 
    tmp_value = path_name_keyword(dir_names)
    index_value_list.append((tmp_index,tmp_value))
    #print index_value_list
    output_strings = []
    for index_value in index_value_list:
        output_strings.append(str(index_value[0]) + ':' + str(index_value[1]))
    return label_name + ' ' + ' '.join(output_strings) +'\n'


def init_feature_index(label_name , url):
    url_portions = urlparse.urlparse(url)
    # path dir name 
    is_endswith_dirsplit = 0
    dir_names = url_portions.path.split('/')
    dir_deepth =  len(dir_names)
    dir_start = 0
    if not url_portions.path.endswith('/'):
        dir_deepth = dir_deepth -1
    if url_portions.path.startswith('/'):
        dir_start = 1
    #     
    for i in range (dir_start,dir_deepth) :
        if  dir_names[i] == '' :
            break
        key = "dir_name_" + dir_names[i]
        if key not in dir_name_index_map.keys() :
            dir_name_index_map[key] = 0
        dir_name_index_map[key] = dir_name_index_map[key] + 1
    
    doc_name = dir_names[len(dir_names)-1]
    if '.' in doc_name:
        doc_name = doc_name[:doc_name.rfind('.')]

    if '201' in doc_name :
        key = "doc_name_" + doc_name 
        if key not in doc_name_index_map.keys() :
            doc_name_index_map[key] = 0
        doc_name_index_map[key] = doc_name_index_map[key] + 1

    ext_name =  dir_names[len(dir_names)-1] 
    if '.' in ext_name:
        ext_name = ext_name[ext_name.rfind('.')+1:]
    else:
        ext_name = ''

    key = "ext_name_" + ext_name 
    if key not in ext_name_index_map.keys() :
        ext_name_index_map[key] = 0
    ext_name_index_map[key] = ext_name_index_map[key] + 1

    query_str = url_portions.query  
    query_params = query_str.split('&')
    max_paramvalue_len = 0
    for query in query_params :
        if '=' not in query:
            continue
        query_name = query[:query.find('=')]
        
        if query_name in infoid_params_name_list:
            query_value =  query[query.find('=')+1 :]
            if query_value.isdigit():
                max_paramvalue_len = len(query_value) if len(query_value) > max_paramvalue_len else  max_paramvalue_len

        key = "param_name_" + query_name 
        if key not in params_name_index_map.keys() :
            params_name_index_map[key] = 0
        params_name_index_map[key] = params_name_index_map[key] + 1
    #--------------
    if max_paramvalue_len not in paravalue_digitlen_score_map.keys():
        paravalue_digitlen_score_map[max_paramvalue_len] = 1
        tmp_result_map [max_paramvalue_len] = []
    else:
        paravalue_digitlen_score_map[max_paramvalue_len] += 1
    tmp_result_map[max_paramvalue_len].append(url)
    # -------------     
    res_name = ''
    if doc_name != '' :
        res_name =  doc_name
    elif len(dir_names)> 1:
        res_name = dir_names[len(dir_names)-2]

    resname_digitlen =  max_continuity_digit(res_name)
    if resname_digitlen not in resname_digitlen_score_map.keys():
        resname_digitlen_score_map[resname_digitlen] = 1
        #tmp_result_map [resname_digitlen] = []
    else:
        resname_digitlen_score_map[resname_digitlen] += 1
    #tmp_result_map[resname_digitlen].append(url)
    # ---------------------

def test_select_feature():
    print 'start '
#    f = open(info_url_file,'r') 
#    f = open(index_url_file,'r') 
    f = open('data/all_subdomain_info_random_20w','r') 
    for l in f :
        l = l.strip()
        init_feature_index(0 , l)
    f.close()
#    print dir_name_index_map
    #print sorted(dir_name_index_map.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)[:100] 
    #print sorted(ext_name_index_map.items(), lambda x, y: cmp(x[1], y[1]), reverse=True) 
    #print sorted(params_name_index_map.items(), lambda x, y: cmp(x[1], y[1]), reverse=True) 
    print '----------------------------------------'
    #print sorted(doc_name_index_map.items(), lambda x, y: cmp(x[1], y[1]), reverse=True) 

    #print sorted(resname_digitlen_score_map.items(), lambda x, y: cmp(x[1], y[1]), reverse=True) 
    print sorted(paravalue_digitlen_score_map.items(), lambda x, y: cmp(x[1], y[1]), reverse=True) 
    of = open('tmp_reslut_map','w')
    for resname_digitlen in  tmp_result_map.keys():
        for url in tmp_result_map[resname_digitlen]:
            of.write(str(resname_digitlen) + '\t' + url + '\n')
    of.close()


def train_and_predict():
    label_0_num = 3000
    label_1_num = 5000
    make_svm_feature_file(index_url_file ,info_url_file, 'feature_file' ,train_label_0_num,train_label_1_num)
    #m =train_predict('feature_file',label_0_num,label_1_num) 
    m = train_predict_random('feature_file', predict_label_0_num ,predict_label_1_num)   
    svm_save_model('url_feature.model', m)
    return m 


def test_file_classifation(url_type,file_name):
    feature_select_fuc = choose_feature_fuc()
    m = svm_load_model('url_feature.model')
    feature_file = 'test_url_feature'
    f = open(feature_file,'w') 
    url_list = []
    feature_list = []
    i_f = open(file_name,'r') 
    for l in i_f :
        url = l.strip()
        url_list.append(url)
        output_line = feature_select_fuc(url_type , url)
        feature_list.append(output_line)
        f.write(output_line)
    f.close()
    i_f.close()
    y, x = svm_read_problem(feature_file)
    p_label, p_acc, p_val = svm_predict(y, x, m)
    n = len(y)
    for i in range(0,n):
        #if p_val[i][0] < 0.5 and p_val[i][0] > -0.5 :
        #if p_val[i][0] < -0.5 :
        print p_label[i],p_val[i],url_list[i],feature_list[i]

    



def test_url_classifation(url_type,url):
    feature_select_fuc = choose_feature_fuc()
    m = svm_load_model('url_feature.model')
    feature_file = 'test_url_feature'
    f = open(feature_file,'w') 
    output_line = feature_select_fuc(url_type , url)
    print output_line
    f.write(output_line)
    f.close()
    y, x = svm_read_problem(feature_file)
    p_label, p_acc, p_val = svm_predict(y, x, m)
    print 'predict label result:',p_label,p_val
   
    

if __name__ == '__main__':
    init()
    test_select_feature()
    app_name = sys.argv[1]
    #make_key_rate('conf/feature_url.txt',line_key = '12_')
    if app_name == 'train': 
        train_and_predict()

    if app_name == 'train_paramter': 
        train_paramters()

    if app_name == 'test_url':
        url_type = sys.argv[2]
        url = sys.argv[3]
        test_url_classifation(url_type,url)

    if app_name == 'test_file':
        url_type = sys.argv[2]
        file_name = sys.argv[3]
        test_file_classifation(url_type,file_name)


