The repository contains several NLG models for couplet and poem generation.

**Model 1: using sequence labeling**  
config file: `config_seq.py`  
start taining: `python train_seq.py`  
some results:
>epoch: 0, loss: 2.032831907272339
>
>input: 兴漫香江红袖舞  
>target: 龙腾粤海碧云飞  
>predict: 风开碧水绿云飞  
>
>input: 河伯捧觞，游观沧海  
>target: 天帝悬车，凭乘风云  
>predict: 山王入月，笑看青山  
>
>input: 清风凝白雪  
>target: 妙舞散红霞  
>predict: 明雨映红云  
>
>input: 木  
>target: 花  
>predict: 花  
>
>input: 讲道德，树新风，遵守法纪  
>target: 学雷锋，做好事，为国利民  
>predict: 为文明，创大气，创兴民生  
>
>...
>
>epoch: 9, loss: 2.0298333168029785
>
>input: 兴漫香江红袖舞  
>target: 龙腾粤海碧云飞  
>predict: 平盈盛水白云飞  
>
>input: 河伯捧觞，游观沧海  
>target: 天帝悬车，凭乘风云  
>predict: 山公作酒，品览春山  
>
>input: 清风凝白雪  
>target: 妙舞散红霞  
>predict: 明月润青风  
>
>input: 木  
>target: 花  
>predict: 油  
>
>input: 讲道德，树新风，遵守法纪  
>target: 学雷锋，做好事，为国利民  
>predict: 修文明，扬正气，弘扬民风  


**Model 2: using VAE**  
config file: `config_vae.py`  
start taining: `python train_vae.py`  
some results:
>epoch: 0, cross_entropy_loss: 63.39518737792969, kld_loss: 1.6401687860488892
>
>result: 不有不山水&lt;SEP&gt;不人不不时
>
>result: 不日秋云里&lt;SEP&gt;春山白上风
>
>result: 高日不不水&lt;SEP&gt;风月白云明
>
>result: 不人不不去&lt;SEP&gt;何是不不知
>
>result: 不日无何去&lt;SEP&gt;风人不不知
>
>epoch: 9, cross_entropy_loss: 49.82965850830078, kld_loss: 10.216238975524902
>
>result: 何来一然意&lt;SEP&gt;一望一相归
>
>result: 寂悠山雨尽&lt;SEP&gt;萧望月相迟
>
>result: 雄峨汉汉阙&lt;SEP&gt;突足为黎功
>
>result: 草色风初尽&lt;SEP&gt;山风夜夜声
>
>result: 一发无人去&lt;SEP&gt;相生君有归
>
>epoch: 19, cross_entropy_loss: 43.603267669677734, kld_loss: 16.23033905029297
>
>result: 临朝苦离思&lt;SEP&gt;何处心纷息
>
>result: 芳树郁苍旭&lt;SEP&gt;红花凝绿枝
>
>result: 登生太区宙&lt;SEP&gt;自迹世生间
>
>result: 一里新松佩&lt;SEP&gt;三年一风来
>
>result: 莫得忘尘时&lt;SEP&gt;天生无寂馀

**Model 3**  
...

Notice:  
The complete couplet dataset can be downloaded from:  
https://github.com/wb14123/couplet-dataset

**requirements**  
pytorch == 1.8.0  
numpy == 1.19.2  
jieba == 0.42.1  
tqdm == 4.51.0

**Reference**  
https://kexue.fm/archives/6270  
https://kexue.fm/archives/5253  
https://kexue.fm/archives/5332  
https://github.com/bojone/vae  
https://github.com/AntixK/PyTorch-VAE