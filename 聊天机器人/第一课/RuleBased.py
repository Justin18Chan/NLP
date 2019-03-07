# -*- coding: utf-8 -*-
"""
基于Rule-Based规则的简单聊天机器人ChatBot
"""

def CharBot_Simple():
    import random
    # 打招呼
    greetings = ['hola', 'hello', 'hi', 'Hi', 'hey!', 'hey']
    # 回复打招呼
    random_greeting = random.choice(greetings)

    # 对于"你怎么样?" 这个问题的回复
    question = ['How are you?', 'How are you doing?']
    # "我很好"
    responses = ['Okay', "I'm fine"]
    # 随机选一个回
    random_response = random.choice(responses)

    # 机器人跑起来
    while True:
        userInput = input(">>>")
        if userInput in greetings:
            print(random_greeting)
        elif userInput in question:
            print(random_response)
            # 除非你说"拜拜"
        elif userInput == 'bye':
            break
        else:
            print("I dit not understand what you said")

#CharBot_Simple()

def ChatBot_intents1():
    """ intent意图机器人, 精准对答, 通过关键词来判断这句话意图是什么?"""
    from nltk import word_tokenize
    import random
    # 打招呼
    greetings = ['hola', 'hello', 'hi', 'Hi', 'hey!', 'hey']
    # 回复打招呼
    random_greeting = random.choice(greetings)
    
    # 对于假期的话题关键词
    question = ['break', 'holiday', 'vacation', 'weekend']
    # 回复假期话题
    responses = ['It was nice! I went to Paris',"Sadly, I just stayed at home"]
    # 随机选一个回
    random_response = random.choice(responses)
    
    # 机器人跑起来
    while True:
        userInput = input(">>>")
        # 清理一下输入，看看都有哪些词
        cleaned_input = word_tokenize(userInput)
        # set.isdisjoint(set)判断两个集合是否有相同元素,有则返回True,没有则返回False
        # 这里,我们比较一下关键词,确定他属于哪个问题
        if not set(cleaned_input).isdisjoint(greetings):
            print(random_greeting)
        elif not set(cleaned_input).isdisjoint(question):
            print(random_response)
            # 除非你说"拜拜"
        elif userInput == "bye":
            break
        else:
            print("I did not understand what you said")

#ChatBot_intents1()
            

def ChatBot_intents3():
    """
    利用Google的API，写一个类似钢铁侠Tony的语音小秘书Jarvis：
    实现从文本到语音的功能.
    我们先来看一个最简单的说话版本。
    利用gTTs(Google Text-to-Speech API), 把文本转化为音频。
    注意: 这里调用google search,需要连接Google服务器,要翻墙.
    """
    from gtts import gTTS
    import os
    tts = gTTS(text="您好, 我是您的私人助手,我叫小辣椒", lang='zh-tw')
    tts.save("hello.mp3")
    os.system("mpg321 hello.mp3")
    print("end")

ChatBot_intents3()

def ChatBot_intents4():
    """
    我们还可以运用Google API读出Jarvis的回复：
    （注意：这里需要你的机器安装几个库 SpeechRecognition, PyAudio 和 PySpeech）
    """
    import speech_recognition as sr
    from time import ctime
    import time
    import os 
    from gtts import gTTS
    import sys
    
    def speak(audioString):
        #讲出来AI的话
        print(audioString)
        tts = gTTS(text=audioString, lang='en')
        tts.save("audio.mps")
        os.system("mpg321 audio.mp3")#mpg321是音频执行命令,没有需要装.
        
    def recordAudio()#
    # 录下来讲的话
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio = r.listen(source)
            
        # 用Google API转化音频
        data = ""
        try:
            data = r.recognize_google(audio)
            print("You said: " + data)
        except sr.UnknownVauleError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e)")
        
        return data
    
    # 自带的对话技能(rules)
    def jarvis():
        while True:
            data = recordAudio()
            if "how are you" in data:
                speak("I am fine")
            if "what time is it" in data:
                speak(ctime())
            if "where is" in data:
                data = data.split(" ")
                location = data[2]
                speak("Hold on Tony, I will show you where " + location + " is.")
                os.system("open -a Safari https://www.google.com/maps/place/" + location + "/&amp;")
                
            if "bye" in data:
                speak("bye bye")
                break
            
    # 初始化
    time.sleep(2)
    speak("Hi Tony, what can I do for you?")
    
    #跑起
    jarvis()
    