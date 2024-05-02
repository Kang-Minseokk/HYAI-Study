from transformers import BertTokenizer, BertConfig, BertModel

tokenizer = BertTokenizer.from_pretrained(
    "beomi/kcbert-base",
    do_lower_case=False
)
pretrained_model_config = BertConfig.from_pretrained(
    "beomi/kcbert-base",
)
model = BertModel.from_pretrained(
    "beomi/kcbert-base",
    config=pretrained_model_config,
)

sentences = ["안녕하세요", "하이!"]
features = tokenizer(
    sentences,
    max_length = 10,
    padding = "max_length",
    truncation = True,
)

print(features)



