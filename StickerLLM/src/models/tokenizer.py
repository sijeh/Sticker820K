from transformers import AutoTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)


SPECIAL_TOKENS = {
    'img_token':'[IMG]',
    'imgend_token':'[/IMG]',
    'ret_token':'[RET]',
    'retend_token':'[/RET]',
    'pret_token':'[PRET]'
}

def build_chatglm_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code = True)

    for token_name,token_content in SPECIAL_TOKENS.items():
        num_added_tokens = tokenizer.add_tokens(token_content)
        token_id = tokenizer(token_content, add_special_tokens=False).input_ids[0]
        setattr(tokenizer,token_name,token_content)
        setattr(tokenizer,token_name+'_id',token_id)
        print(f'Add new token {token_name}: {token_id}')


    return tokenizer



    
