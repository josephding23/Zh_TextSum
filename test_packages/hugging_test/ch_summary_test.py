from transformers import pipeline
from util.hugging import get_local_path
from transformers import MBartTokenizer, BartModel, AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained(get_local_path("facebook/bart-large-cnn"))
model = AutoModelWithLMHead.from_pretrained(get_local_path("facebook/bart-large-cnn"))

summarizer = pipeline("summarization",
                      model=model,
                      tokenizer=tokenizer)

ARTICLE = """ 
从6月底到7月，美国媒体和民意机构进行了一系列全国范围内的民意调查。结果显示，拜登对特朗普一路保持优势，一些民调还出现2位数的优势。特朗普2016年赢下的几场关键竞选和州，拜登也都保持领先。
拜登可能会对着民调露出微笑，但内心不应该有任何放松。因为 4年前，民调显示希拉里的优势也很大。
对于2016年的失败，民调专家表示他们已经学到了教训，但民调需要的是微小的调整，而不是根本的改革。2016年的民意调查和最终结果不一致，这样很多选民失望，但并不意味着大规模民意不再有效。
为什么会出现2016年那样的不一致，民调专家总结：数字并不是真正的安全，因为选举的最后一分钟可能让所有人的预料出错。
2016年确实有预测成功的时候，比如希拉里在大多数州获胜。可民调没能正确反映美国西北部地区选民的情绪，正是这种情绪让特朗普在选举团中获得了优势。在威斯康辛州、密歇根州和宾夕法尼亚州的104项民调中，101项预测希拉里获胜，2项打平，1项预测特朗普微弱获胜。最终是特朗普将这三个州全部拿下。
于是民调专家提出很多改进意见，比如应该把高中学历的选民纳入调查对象范围，但这部分人通常也不愿意参加民调；把独立候选人加入民调中；更加精确地确定潜在选民等。
一位新泽西州的普通选民道出了民调背后的哲学：选民就像冰山，你永远看到的是冰山一角，看不到整座冰山，你不知道他们会投给谁。特朗普支持者认为，总统现在民调落后拜登，其实低估了总统的力量，但特朗普支持者不愿意告诉民调机构自己的观点，“人们倾向于保留自己的观点”。拜登的支持者理解特朗普支持者为什么隐藏自己的观点，
“承认自己支持特朗普，可能会让他们感到尴尬”。对于拜登领先的民调，“我觉得，他们改进了调查方式，有更准确的结果”。
除了前面的改进意见，民调专家也重新思考民调的定义和数据背后的含义：民调是关于现在发生的事情，以及人们如何认识，并不是关于选举团的最终结果”。比如在6月的一次全国性民调，这次民调的重点是选民对特朗普的看法，因为很多人还不是很了解拜登，随着公众越来越了解拜登，这种情况可能会在10月份发生改变。
和2016年一样，2020年的民调再次显示特朗普在密歇根州、宾夕法尼亚州和威斯康星州败北。
共和党全国委员会主席麦克丹尼尔对民调很不在乎，“这些民调现在不重要，我们2016年做了150次民调都现实特朗普会输，重要的是选举日”。但特朗普似乎很看重民调的数字，4年间，主流媒体ABC，NBC，CNN和纽约时报的民调都被他骂个遍。
距离11月3日的大选日还剩下112天，对于特朗普和拜登来说，严密准确的竞选策略和执行力比民调的数字，可能更让他们安心。
          """

# print(len(ARTICLE))

# inputs = tokenizer.encode("summarize: " + ARTICLE, return_tensors="pt", max_length=512)
# outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

# print(outputs)
summary = summarizer(ARTICLE, max_length=150, min_length=40, do_sample=False)
print(len(ARTICLE), len(summary[0]['summary_text']))
