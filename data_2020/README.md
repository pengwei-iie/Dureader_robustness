# DuReader<sub>robust</sub> Dataset
##Files
>**train.json:** the training set that contains around 15K samples. 

>**dev.json:** the development set that contains around 1.4K samples. 

>**evaluate.py:** the evaluation script.

Please refer to the paper of DuReader<sub>robust</sub> for more details. 

##Data Format

Here is an example of a data sample:
```
{
    "data": [
        {
            "paragraphs": [
                {
                    "qas": [
                        {
                            "question": "韩国全称", 
                            "id": "a7eec8cf0c55077e667e0d85b45a6b34", 
                            "answers": [
                                {
                                    "text": "大韩民国", 
                                    "answer_start": 5
                                }
                            ]
                        }
                    ], 
                    "context": "韩国全称“大韩民国”，位于朝鲜半岛南部，隔“三八线”与朝鲜民主主义人民共和国相邻，面积9.93万平方公理，南北长约500公里，东西宽约250公里，东濒日本海，西临黄海 ，东南与日本隔海相望。 韩国的地形特点是山地多，平原少，海岸线长而曲折。韩国四 季分明，气候温和、湿润。目前韩国主要政党包括执政的新千年民主党和在野的大国家党、自由民主联盟等，大国家党为韩国会内的第一大党。韩国首都为汉城，全国设有1个特别市（汉城市）、6个广域市（釜山市、大邱市、仁川市、光州市、大田市、蔚山市）、9个道（京畿道、江源道、忠清北道、忠清南道、全罗北道、全罗南道、庆尚北道、庆尚南道、济州道）。海岸线全长5,259公里，主要港口有釜山、仁川、浦项、蔚山、光阳等。"
                }
            ],
            "title": ""
        }
    ]
}
```
Basically, the data format of DuReader<sub>robust</sub> dataset is compatible with the one of SQuAD. Here, the field of **"context"** contains the text of the context in the given sample. The field of **"question"** contains the text of the question that is asked about the **"context"**, and **"id"** is the uniq id for each question. The sub-field **"text"** of **"answers"** contains the text of a human annotated answer, and the sub-field **"answer_start"** denotes the start position of the human annotated answer in the origional **"context"**. 




##Submission Requirement

To submit your results and get it shown on the leaderboard, we require the participants to submit their results in JSON format, where the ID and the corresponding extracted answer for each question is a (key, value) pair. A sample is as follows:
```
{
    "a7eec8cf0c55077e667e0d85b45a6b34": "大韩民国", 
    "0f5fd3b9c571150ee65198766fde0e17": "暹", 
    "365148fda3b5173d5e77c4befc0616ff": "AP通用打野符文", 
    "80fbb0ffd67edcddc62094cd225d2ab3": "百度地图", 
    "a5d6003f37c985897f63bc939e21d649": "半年", 
    "336692113086534a8cea1707accb2377": "5900点券", 
    "fac159799ead297fd1e0e91ee0dc5714": "不超过20分钟", 
    "7ff4704394d934968e73005b914692b6": "奕剑", 
    "1482e1ef44b44567c39897cc374d7897": "乔纳森·斯威夫特", 
    "4d38efd249e8ec129672ff688ce80e46": "阿联酋航空", 
    "24f2e66dc484074f38437a342bd344a4": "鲁花", 
    "4589f098229bf7f51e1d824d0f3dc07c": "一岁一个月", 
    "7036f27487d5a2d152dcd50a660fc81e": "20元", 
    "14952655aee0e0bc9408c488f473efee": "半音", 
    "340039ee594aa191bb8eb60dc7e3005c": "11岁3个月"
}
```

