## 聊天机器人ChatBot

1. 生成模型 Generative Model  
必读paper: A Neural Conversation Model  
训练两组对话,一个是电影对白,一个是IT维护对话.  

2. 聊天机器人库ChatterBot  
pip install chatterbot  

3. 知识框架  

Open Domain     |    impossible                    General AI[Hardest]  
Closed Domain   |    Rules-Based[Easiest]          Smart Machine [Hard]  
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++     
+++++++++++++++++++Retrieval Based                    Generative Based    

这里几个概念:  
Open Domain: 可以理解为广域, 即可以随便聊,什么领域都可以.  
Closed Domain: 专业域, 即只可以聊某个专业领域.  
Retrieval Based: 理解为需要人为设置.  
Generative Based: 不需要人为设置,直接用数据喂就可以.  

Siri的做法： General AI
比如查找附近的美食餐厅： 借助强大的搜索比如google，百度； 使用搜索引擎API进行搜索，将得到的比如大众点评API结果返回，并调用地图API计算距离。将最后结果返回Siri。  
这样做的好处是：1.可以实现广域搜索。 2. 大大减少了前期训练。 3. 可以得到有一定基础的结果。  

4. 挑战---语言语境  
相关paper：    
Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models（Lulian et al., 2015）https://arxiv.org/abs/1507.04808  
Attention with Intention for a Neural Network Conversation Model (Yao, 2015) https://arxiv.org/abs/1510.08565  
5. 统一的语言个性  
A Persona-Based Neural Conversation Model (Li et al., 2016) https://arxiv.org/abs/1603.06155  
