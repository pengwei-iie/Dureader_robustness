from translate import Translator
import numpy

#在任何两种语言之间，中文翻译成英文
translator = Translator(from_lang="chinese", to_lang="english")
translation = translator.translate("床前明月光，疑是地上霜;举头望明月,低头思故乡")
print(translation)
