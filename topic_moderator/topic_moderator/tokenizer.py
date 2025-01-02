from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruBert-large",
                                          do_lower_case=True)
