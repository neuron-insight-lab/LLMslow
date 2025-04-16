
from .opt_utils import load_model_and_tokenizer
from .opt_utils import get_loss
from .opt_utils import get_gradients
from .opt_utils import get_all_losses
from .opt_utils import sample_control
from .opt_utils import get_filtered_cands

from .string_utils import read_data
from .string_utils import get_chat_prompt
from .string_utils import generate_str
from .string_utils import get_nonascii_toks
from .string_utils import test_suffix
from .string_utils import SuffixManager

MODEL_PATHS = {
    'llama2': '/root/autodl-tmp/models/models/Llama-2-7b-chat-hf',     # meta-llama/Llama-2-7b-chat-hf
    'llama3': '/root/autodl-tmp/models/models/Llama-3-8B-Instruct',   # meta-llama/Meta-Llama-3.1-8B-Instruct
    'mistral': '/root/autodl-tmp/models/models/Mistral-7B-Instruct-v0.3',      # mistralai/Mistral-7B-Instruct-v0.3
    'qwen2.5': '/root/autodl-tmp/models/models/Qwen2.5-7B-Instruct',     #Qwen/Qwen2.5-7B-Instruct
    'glm4': '/root/autodl-tmp/models/models/glm-4-9b-chat',    #THUDM/glm-4-9b-chat-hf
    'phi': '/root/autodl-tmp/models/models/Phi-3.5-mini-instruct',   #microsoft/Phi-3.5-mini-instruct
    'llama3-3b': '/root/autodl-tmp/models/models/Llama-3.2-3B-Instruct', #meta-llama/Llama-3.2-3B-Instruct
    'qwen2.5-1.5b': '/root/autodl-tmp/models/models/Qwen2.5-1.5B-Instruct',     #Qwen/Qwen2.5-1.5B-Instruct
    # 'deepseek-8b': '/root/autodl-tmp/models/models/DeepSeek-R1-Distill-Llama-8B'       #deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    # 'vicuna': '/root/autodl-tmp/models/models/vicuna-7b-v1.5',      #lmsys/vicuna-7b-v1.5
}