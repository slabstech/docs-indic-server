from vllm import LLM
from vllm.sampling_params import SamplingParams
from vllm.inputs.data import TokensPrompt
import requests
from PIL import Image
from io import BytesIO
from vllm.multimodal import MultiModalDataBuiltins

from mistral_common.protocol.instruct.messages import TextChunk, ImageURLChunk

model_name = "mistralai/Pixtral-12B-Base-2409"
sampling_params = SamplingParams(max_tokens=8192)

llm = LLM(model=model_name, tokenizer_mode="mistral")

url = "https://huggingface.co/datasets/patrickvonplaten/random_img/resolve/main/yosemite.png"
response = requests.get(url)
image = Image.open(BytesIO(response.content))

prompt = "The image shows a"

user_content = [ImageURLChunk(image_url=url), TextChunk(text=prompt)]

tokenizer = llm.llm_engine.tokenizer.tokenizer.mistral.instruct_tokenizer
tokens, _ = tokenizer.encode_user_content(user_content, False)

prompt = TokensPrompt(
    prompt_token_ids=tokens, multi_modal_data=MultiModalDataBuiltins(image=[image])
)
outputs = llm.generate(prompt, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
# ' view of a river flowing through the landscape, with prominent rock formations visible on either side of the river. The scene is captured using the UWA 14-24mm zoom lens, which provides a wide-angle perspective,
# allowing for a comprehensive view of the surroundings. The photo is credited to Greg Dowdy.
