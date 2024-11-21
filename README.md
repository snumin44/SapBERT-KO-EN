# 🍊 SapBERT-KO-EN

- 한국어 모델을 이용한 **SapBERT**(Self-alignment pretraining for BERT)입니다.
- 한·영 의료 용어 사전인 KOSTOM을 사용해 한국어 용어와 영어 용어 정렬합니다.
- 참고: [SapBERT](https://aclanthology.org/2021.naacl-main.334.pdf), [Original Code](https://github.com/cambridgeltl/sapbert) 

&nbsp;

## 1. SapBERT-KO-EN
- SapBERT는 수많은 **의료 동의어**를 동일한 의미로 처리하기 위해 의료 용어를 사전학습하는 방법론입니다.
- Multi-Similarity Loss를 이용해 **동일한 의료 코드**를 지닌 용어 간의 유사도를 키우는 방식으로 학습합니다. 


