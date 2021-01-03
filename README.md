### 运行流程  
###### 一、数据处理：（跑一次即可）
* 将trainset、devset等数据放在data文件里 (data下的trainset、devset有部份数据，可以换成全部数据。)
* 到handle_data目录下运行 sh run.sh --para_extraction, 便会将处理后的数据放在extracted下的对应文件夹里
###### 二、制作dataset：（跑一次即可）
* 到dataset目录下运行两次 python3 run_squad.py，分别生成train.data与dev.data,第一次运行结束后要修改run_squad.py的参数，具体做法run_squad.py末尾有具体说明
###### 三、训练：
* 到root下运行 python3 train.py，边训练边验证
###### 四、测试:
* 到predict目录下运行 python3 util.py (测试集太多，也可以在该文件里将路径改为验证集，默认为验证集路径)（跑一次即可）
* 运行 python3 predicting.py
* 到metric目录下， 运行 python3 mrc_eval.py predicts.json ref.json v1 即可

#### 中英文翻译：是的
需要安装包--translate，pip install translate,这里的translate包是微软的,但是有翻译数量的限制
解决方法二：可以采用科大讯飞官网的翻译工具，按照官方给的API即可使用。优点是可翻译的数量较多，需要自行注册。
可参考：https://www.xfyun.cn/services/xftrans?ch=btr&b_scene_zt=1
demo下载：https://www.xfyun.cn/doc/nlp/xftrans/API.html#%E6%8E%A5%E5%8F%A3demo