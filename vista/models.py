import os
import re
import torch
from PIL import Image

# --------------------------
# CONFIGURATION
# --------------------------
# Model paths can be overridden via environment variables.
# HuggingFace model IDs are used as defaults where available.
COGVLM_MODEL_PATH = os.environ.get("VISTA_COGVLM_PATH", "zai-org/cogvlm-grounding-generalist-hf")
COGVLM_TOKENIZER_PATH = os.environ.get("VISTA_COGVLM_TOKENIZER_PATH", "lmsys/vicuna-7b-v1.5")
SHIKRA_MODEL_PATH = os.environ.get("VISTA_SHIKRA_PATH", "shikras/shikra-7b")
FERRET_MODEL_PATH = os.environ.get("VISTA_FERRET_PATH", "ml-ferret/ferret-7b-v1-3")
QWEN3VL_MODEL_PATH = os.environ.get("VISTA_QWEN3VL_PATH", "Qwen/Qwen3-VL-8B-Instruct")
MIMOVL_MODEL_PATH = os.environ.get("VISTA_MIMOVL_PATH", "XiaomiMiMo/MiMo-VL-7B-RL")
INTERNVL25_MODEL_PATH = os.environ.get("VISTA_INTERNVL25_PATH", "OpenGVLab/InternVL2_5-8B")
QWENVL_MODEL_PATH = os.environ.get("VISTA_QWENVL_PATH", "Qwen/Qwen-VL-Chat")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GDINO_MODEL_PATH = os.environ.get("VISTA_GDINO_PATH", "IDEA-Research/grounding-dino-tiny")
MINIGPT4_REPO = os.environ.get("VISTA_MINIGPT4_REPO", "MiniGPT-4")                 # clone of Vision-CAIR/MiniGPT-4
MINIGPTV2_LLAMA_PATH = os.environ.get("VISTA_LLAMA2_CHAT_PATH", "meta-llama/Llama-2-7b-chat-hf")
MINIGPTV2_CKPT = os.environ.get("VISTA_MINIGPTV2_CKPT", "checkpoints/minigptv2_checkpoint.pth")  # stage-3 checkpoint
# 1 (default) = official eval config: Llama-2 loaded 8-bit (bitsandbytes). 0 = the repo's own fp16 path (low_resource=False).
MINIGPTV2_LOW_RESOURCE = os.environ.get("VISTA_MINIGPTV2_LOW_RESOURCE", "1") != "0"
DEVICE = "cuda"

class FerretSingleSample:
    """
    A class to handle loading the Ferret model and running single-sample inference.
    Requires ml-ferret to be installed: https://github.com/apple/ml-ferret
    """
    def __init__(self):
        try:
            from ml_ferret.ferret.model import FERRETLlamaForCausalLM
            from ml_ferret.ferret.conversation import conv_templates
            from ml_ferret.ferret.mm_utils import process_images, tokenizer_image_token
            from ml_ferret.ferret.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
            from transformers import AutoTokenizer, CLIPImageProcessor
        except ImportError:
            raise ImportError(
                "Ferret dependencies not found. Please clone and install ml-ferret: "
                "https://github.com/apple/ml-ferret"
            )

        # Store references for use in run_inference
        self._conv_templates = conv_templates
        self._process_images = process_images
        self._tokenizer_image_token = tokenizer_image_token
        self._IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self._DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN

        print(f"Loading Ferret model from {FERRET_MODEL_PATH}...")
        self.tokenizer = AutoTokenizer.from_pretrained(FERRET_MODEL_PATH, use_fast=False)

        self.model = FERRETLlamaForCausalLM.from_pretrained(
            FERRET_MODEL_PATH,
            torch_dtype=torch.float16,
        ).to(DEVICE)

        print("Loading image processor...")
        vision_tower_name = self.model.config.mm_vision_tower
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name, torch_dtype=torch.float16)

        # Explicitly load vision tower weights and move to the correct device
        print("Explicitly loading vision tower weights...")
        self.model.get_vision_tower().load_model()
        self.model.get_vision_tower().to(device=DEVICE, dtype=torch.float16)
        
        self.model.eval()
        print("Ferret model loaded successfully!")

    def run_inference(self, image: Image.Image, question: str):
        """
        Runs inference on a single image and question.

        Returns:
            tuple: (text_output, boxes, query, response) to match the evaluation script's interface.
        """
        # 1. Prepare the conversation prompt
        conv = self._conv_templates["ferret_v1"].copy()
        text_prompt = "What is the location of " + question + "?" + f'\n{self._DEFAULT_IMAGE_TOKEN}'
        conv.append_message(conv.roles[0], text_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # 2. Process image and tokenize prompt
        image_tensor = self._process_images([image.convert('RGB')], self.image_processor, self.model.config).to(DEVICE, dtype=torch.float16)
        input_ids = self._tokenizer_image_token(prompt, self.tokenizer, self._IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(DEVICE)

        # 3. Generate response
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=64,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # 4. Decode the output
        response = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

        # 5. Parse bounding boxes from the response
        text_output = response
        boxes = []
        # Ferret outputs boxes like '... a bird at [260, 335, 405, 513].'
        match = re.search(r'\[(\d+,\s*\d+,\s*\d+,\s*\d+)\]', response)
        if match:
            try:
                coords_str = match.group(1).split(',')
                # Coords are already in the 0-1000 range, as expected by the eval script
                coords = [int(c.strip()) for c in coords_str]
                if len(coords) == 4:
                    box_tensor = torch.tensor(coords, device=DEVICE).unsqueeze(0)
                    boxes.append(box_tensor)
            except (ValueError, IndexError):
                print("Could not parse coordinates from Ferret response.")

        return text_output, boxes, text_prompt, response

class ShikraSingleSample:
    """
    Wrapper for Shikra model inference.
    Requires shikra to be installed: https://github.com/shikras/shikra
    """
    def __init__(self, model_path=SHIKRA_MODEL_PATH, load_in_8bit=False):
        try:
            from mmengine import Config
            from transformers import BitsAndBytesConfig
            from shikra.mllm.dataset.process_function import PlainBoxFormatter
            from shikra.mllm.dataset.builder import prepare_interactive
            from shikra.mllm.models.builder.build_shikra import load_pretrained_shikra
            from shikra.mllm.dataset.utils.transform import expand2square
        except ImportError:
            raise ImportError(
                "Shikra dependencies not found. Please clone and install shikra: "
                "https://github.com/shikras/shikra"
            )

        self._prepare_interactive = prepare_interactive
        self._expand2square = expand2square

        model_args = Config(dict(
            type='shikra',
            version='v1',
            cache_dir=None,
            model_name_or_path=model_path,
            vision_tower=r'openai/clip-vit-large-patch14',
            pretrain_mm_mlp_adapter=None,
            mm_vision_select_layer=-2,
            model_max_length=512,
            freeze_backbone=False,
            tune_mm_mlp_adapter=False,
            freeze_mm_mlp_adapter=False,
            is_multimodal=True,
            sep_image_conv_front=False,
            image_token_len=256,
            mm_use_im_start_end=True,
            target_processor=dict(boxes=dict(type='PlainBoxFormatter')),
            process_func_args=dict(
                conv=dict(type='ShikraConvProcess'),
                target=dict(type='BoxFormatProcess'),
                text=dict(type='ShikraTextProcess'),
                image=dict(type='ShikraImageProcessor'),
            ),
            conv_args=dict(
                conv_template='vicuna_v1.1',
                transforms=dict(type='Expand2square'),
                tokenize_kwargs=dict(truncation_size=None),
            ),
        ))
        
        training_args = Config(dict(bf16=False, fp16=True, device=DEVICE, fsdp=None))
        
        quantization_kwargs = {}
        if load_in_8bit:
            quantization_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)

        print("Loading Shikra model...")
        self.model, self.preprocessor = load_pretrained_shikra(model_args, training_args, **quantization_kwargs)
        
        # Move model components to the correct device and dtype
        if not getattr(self.model, 'is_quantized', False):
            self.model.to(dtype=torch.float16, device=torch.device(DEVICE))
        if hasattr(self.model.model, 'vision_tower') and not getattr(self.model.model.vision_tower[0], 'is_quantized', False):
            self.model.model.vision_tower[0].to(dtype=torch.float16, device=torch.device(DEVICE))
        
        self.preprocessor['target'] = {'boxes': PlainBoxFormatter()}
        self.tokenizer = self.preprocessor['text']
        self.model_args = model_args
        print("Shikra model loaded successfully.")

    def run_inference(self, image: Image.Image, question: str):
        """
        Runs inference on a single image and question, returning the parsed bounding box.
        """
        prompt = f'Would you kindly provide the coordinates of {question} located in the picture?'
        processed_image = self._expand2square(image.convert("RGB"))

        # Prepare conversational input
        ds = self._prepare_interactive(self.model_args, self.preprocessor)
        ds.set_image(processed_image)
        ds.append_message(role=ds.roles[0], message=prompt)
        
        model_inputs = ds.to_model_input()
        for k, v in model_inputs.items():
            if torch.is_tensor(v):
                model_inputs[k] = v.to(DEVICE)
        
        if 'images' in model_inputs and model_inputs['images'] is not None:
            model_inputs['images'] = model_inputs['images'].to(dtype=torch.float16)

        # Define generation arguments
        gen_kwargs = dict(
            use_cache=True,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=64,
        )

        # Run model generation
        with torch.inference_mode():
            with torch.autocast(dtype=torch.float16, device_type='cuda'):
                output_ids = self.model.generate(**model_inputs, **gen_kwargs)

        # Decode and parse the response
        input_token_len = model_inputs['input_ids'].shape[-1]
        response_ids = output_ids[:, input_token_len:]
        response = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0].strip()

        text_output = response
        boxes = []
        
        # Parse the bounding box from the response string, e.g., "Here it is [100, 200, 300, 400]"
        match = re.search(r'\[(\d+\.?\d*,\s*\d+\.?\d*,\s*\d+\.?\d*,\s*\d+\.?\d*)\]', response)
        if match:
            try:
                coords_str = match.group(1).split(',')

                # Multiply by 1000 because Shikra output coords in 0..1 range.
                # Rest of the script uses 0..1000 range
                coords = [(float(c.strip()) * 1000) for c in coords_str]
                if len(coords) == 4:
                    box_tensor = torch.tensor(coords, device=DEVICE).unsqueeze(0)
                    boxes.append(box_tensor)
            except (ValueError, IndexError):
                print("Could not parse coordinates from Shikra response.")

        return text_output, boxes, prompt, response

class CogVLMSingleSample:
    def __init__(self):
        from transformers import AutoModelForCausalLM, LlamaTokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            COGVLM_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(DEVICE).eval()
        self.tokenizer = LlamaTokenizer.from_pretrained(COGVLM_TOKENIZER_PATH)


    def run_inference(self, image, question: str):
        question = question.strip().replace('\\', '')
        query = f'Detect and provide coordinates for "{question}"'

        input_by_model = self.model.build_conversation_input_ids(
            self.tokenizer,
            query=query,
            history=[],
            images=[image]
        )

        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(torch.bfloat16)]],
        }

        gen_kwargs = {"max_new_tokens": 1024, "do_sample": False}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        text_output = response
        boxes = []
        
        match = re.search(r'\[\[(\d+\.?\d*,\d+\.?\d*,\d+\.?\d*,\d+\.?\d*)\]\]', response)
        if match:
            try:
                # Extract the numbers, split by comma, and convert to floats
                coords = [int(c) for c in match.group(1).split(',')]
                if len(coords) == 4:
                    box_tensor = torch.tensor(coords, device=DEVICE).unsqueeze(0)
                    boxes.append(box_tensor)
            except (ValueError, IndexError):
                print("Could not parse coordinates from model response.")
                pass # Failed to parse, boxes remains empty

        return text_output, boxes, query, response


class Qwen3VLSingleSample:
    """Wrapper for Qwen3-VL-8B-Instruct model.

    Qwen3-VL outputs coordinates in 1000x1000 normalized grid,
    matching our existing pipeline exactly.
    """

    def __init__(self, model_path=QWEN3VL_MODEL_PATH, device=DEVICE):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise ImportError(
                "qwen-vl-utils not found. Install with: pip install qwen-vl-utils"
            )
        self._process_vision_info = process_vision_info

        print(f"Loading Qwen3-VL model from {model_path}...")
        self.device = device

        self.processor = AutoProcessor.from_pretrained(model_path)

        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            ).to(device)
            print("Using flash_attention_2")
        except Exception:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).to(device)
            print("Using sdpa attention (flash_attention_2 not available)")

        self.model.eval()
        print("Qwen3-VL model loaded successfully!")

    def run_inference(self, image: Image.Image, question: str):
        """
        Runs inference on a single image and question.

        Returns:
            tuple: (text_output, boxes, query, response) to match the evaluation script's interface.
            boxes: list of tensors with [x1, y1, x2, y2] in 0-1000 scale
        """
        prompt = f"Locate {question} in this image and provide the bounding box coordinates."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        boxes = self._parse_boxes(response)
        return response, boxes, prompt, response

    def _parse_boxes(self, text: str):
        """Extract bounding boxes from model output.

        Qwen3-VL outputs coords in 1000x1000 grid natively.
        """
        boxes = []
        pattern = r'[\[\(](\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)[\]\)]'
        matches = re.findall(pattern, text)
        for match in matches:
            coords = [int(x) for x in match]
            box_tensor = torch.tensor(coords, device=self.device).unsqueeze(0)
            boxes.append(box_tensor)
        return boxes


class MiMoVLSingleSample:
    """Wrapper for MiMo-VL-7B-RL model.

    MiMo-VL uses Qwen2.5-VL architecture and outputs coordinates
    in PIXEL space, converted to 0-1000 normalized for the pipeline.
    """

    def __init__(self, model_path=MIMOVL_MODEL_PATH, device=DEVICE):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise ImportError(
                "qwen-vl-utils not found. Install with: pip install qwen-vl-utils"
            )
        self._process_vision_info = process_vision_info

        print(f"Loading MiMo-VL model from {model_path}...")
        self.device = device

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            ).to(device)
            print("Using flash_attention_2")
        except Exception:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                trust_remote_code=True,
            ).to(device)
            print("Using sdpa attention (flash_attention_2 not available)")

        self.model.eval()
        print("MiMo-VL model loaded successfully!")

    def run_inference(self, image: Image.Image, question: str):
        img_w, img_h = image.size
        prompt = f"Locate {question} in this image and provide the bounding box coordinates. /no_think"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        boxes = self._parse_boxes(response, img_w, img_h)
        return response, boxes, prompt, response

    def _parse_boxes(self, text: str, img_w: int, img_h: int):
        """Extract bounding boxes from model output.

        MiMo-VL outputs coords in PIXEL space, not 0-1000 normalized.
        We convert them to 0-1000 normalized to match the pipeline.
        """
        boxes = []
        pattern = r'[\[\(](\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)[\]\)]'
        matches = re.findall(pattern, text)
        for match in matches:
            x1, y1, x2, y2 = [int(x) for x in match]
            x1_norm = x1 / img_w * 1000
            y1_norm = y1 / img_h * 1000
            x2_norm = x2 / img_w * 1000
            y2_norm = y2 / img_h * 1000
            coords = [x1_norm, y1_norm, x2_norm, y2_norm]
            box_tensor = torch.tensor(coords, device=self.device).unsqueeze(0)
            boxes.append(box_tensor)
        return boxes


class InternVL25SingleSample:
    """Wrapper for InternVL2.5-8B (InternViT-300M + InternLM2.5-7B).

    Preprocessing and the grounding prompt follow the official model card:
    dynamic 448px tiling (max 12 tiles + thumbnail), ImageNet normalisation,
    and the `<ref>...</ref>` referring-expression prompt. InternVL2.5 answers with
    `<box>[[x1, y1, x2, y2]]</box>` on a 1000x1000 normalised grid, matching our pipeline.
    """

    def __init__(self, model_path=INTERNVL25_MODEL_PATH, device=DEVICE, max_num=12):
        from transformers import AutoModel, AutoTokenizer
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        print(f"Loading InternVL2.5 model from {model_path}...")
        self.device = device
        self.max_num = max_num
        self.input_size = 448
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((self.input_size, self.input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        try:
            self.model = AutoModel.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                use_flash_attn=True, trust_remote_code=True,
            ).eval().to(device)
            print("Using flash attention")
        except Exception as e:
            print(f"flash attention unavailable ({type(e).__name__}); falling back")
            self.model = AutoModel.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                use_flash_attn=False, trust_remote_code=True,
            ).eval().to(device)
        self.generation_config = dict(max_new_tokens=128, do_sample=False)
        print("InternVL2.5 model loaded successfully!")

    # --- preprocessing (from the InternVL2.5 model card) ---
    @staticmethod
    def _closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff, best_ratio = float("inf"), (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff, best_ratio = ratio_diff, ratio
            elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(self, image, min_num=1, use_thumbnail=True):
        image_size = self.input_size
        w, h = image.size
        target_ratios = sorted(
            {(i, j) for n in range(min_num, self.max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
             if min_num <= i * j <= self.max_num},
            key=lambda x: x[0] * x[1])
        ratio = self._closest_aspect_ratio(w / h, target_ratios, w, h, image_size)
        tw, th = image_size * ratio[0], image_size * ratio[1]
        blocks = ratio[0] * ratio[1]
        resized = image.resize((tw, th))
        tiles = []
        for i in range(blocks):
            box = ((i % (tw // image_size)) * image_size, (i // (tw // image_size)) * image_size,
                   ((i % (tw // image_size)) + 1) * image_size, ((i // (tw // image_size)) + 1) * image_size)
            tiles.append(resized.crop(box))
        if use_thumbnail and len(tiles) != 1:
            tiles.append(image.resize((image_size, image_size)))
        return tiles

    def run_inference(self, image: Image.Image, question: str):
        """
        Runs inference on a single image and question.

        Returns:
            tuple: (text_output, boxes, query, response) to match the evaluation script's interface.
            boxes: list of tensors with [x1, y1, x2, y2] in 0-1000 scale
        """
        tiles = self._dynamic_preprocess(image)
        pixel_values = torch.stack([self.transform(t) for t in tiles]).to(torch.bfloat16).to(self.device)
        prompt = f"<image>\nPlease provide the bounding box coordinate of the region this sentence describes: <ref>{question}</ref>"
        with torch.no_grad():
            response = self.model.chat(self.tokenizer, pixel_values, prompt, self.generation_config)
        boxes = self._parse_boxes(response)
        return response, boxes, prompt, response

    def _parse_boxes(self, text: str):
        """Extract boxes from `<box>[[x1, y1, x2, y2]]</box>` (0-1000 grid); falls back to any 4-number list."""
        boxes = []
        pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
        for match in re.findall(pattern, text):
            coords = [int(x) for x in match]
            boxes.append(torch.tensor(coords, device=self.device).unsqueeze(0))
        return boxes


class QwenVLChatSingleSample:
    """Wrapper for Qwen-VL-Chat (ViT-bigG + Qwen-7B).

    Uses the model's remote code and the grounding pattern from its model card: the
    referring expression is wrapped in `<ref>...</ref>` and the model answers
    `<ref>...</ref><box>(x1,y1),(x2,y2)</box>` on a 1000x1000 normalised grid.
    """

    def __init__(self, model_path=QWENVL_MODEL_PATH, device=DEVICE):
        import tempfile
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading Qwen-VL-Chat model from {model_path}...")
        self.device = device
        self.tmpdir = tempfile.mkdtemp(prefix="qwenvl_")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=device, trust_remote_code=True, bf16=True,
        ).eval()
        self.model.generation_config.do_sample = False
        print("Qwen-VL-Chat model loaded successfully!")

    def run_inference(self, image: Image.Image, question: str):
        """
        Runs inference on a single image and question.

        Returns:
            tuple: (text_output, boxes, query, response) to match the evaluation script's interface.
            boxes: list of tensors with [x1, y1, x2, y2] in 0-1000 scale
        """
        frame_path = os.path.join(self.tmpdir, "frame.jpg")
        image.convert("RGB").save(frame_path, quality=95)
        prompt = f"<ref>{question}</ref>"
        query = self.tokenizer.from_list_format([{"image": frame_path}, {"text": prompt}])
        with torch.no_grad():
            response, _ = self.model.chat(self.tokenizer, query=query, history=None)
        boxes = self._parse_boxes(response)
        return response, boxes, prompt, response

    def _parse_boxes(self, text: str):
        """Extract boxes from `<box>(x1,y1),(x2,y2)</box>` (0-1000 grid)."""
        boxes = []
        pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)'
        for match in re.findall(pattern, text):
            coords = [int(x) for x in match]
            boxes.append(torch.tensor(coords, device=self.device).unsqueeze(0))
        return boxes


class GroundingDINOSingleSample:
    """Wrapper for Grounding DINO (Swin-T backbone, BERT text encoder) via transformers.

    Returns the single query box with the highest text-matching score. Scores and boxes are read directly from the model
    outputs (sigmoid of the token logits, max over tokens; boxes as normalised cx, cy, w, h), so no post-processing API
    is involved.
    """

    def __init__(self, model_path=GDINO_MODEL_PATH, device=DEVICE):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        print(f"Loading Grounding DINO from {model_path}...")
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(device).eval()
        print("Grounding DINO loaded successfully!")

    def run_inference(self, image: Image.Image, question: str):
        """
        Runs inference on a single image and question.

        Returns:
            tuple: (text_output, boxes, query, response) to match the evaluation script's interface.
            boxes: list of tensors with [x1, y1, x2, y2] in 0-1000 scale
        """
        text = question.strip().lower()
        if not text.endswith("."):
            text += "."  # Grounding DINO expects period-terminated phrases
        inputs = self.processor(images=image.convert("RGB"), text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        scores = out.logits.sigmoid()[0].max(dim=-1).values          # (num_queries,)
        best = int(scores.argmax())
        cx, cy, w, h = out.pred_boxes[0, best].tolist()               # normalised to [0, 1]
        box = [max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(1.0, cx + w / 2), min(1.0, cy + h / 2)]
        coords = [int(round(v * 1000)) for v in box]
        response = f"score={float(scores[best]):.4f} box={coords}"
        return response, [torch.tensor(coords, device=self.device).unsqueeze(0)], text, response


class MiniGPTv2SingleSample:
    """Wrapper for MiniGPT-v2 (EVA-CLIP ViT-G/14 + Llama-2-7B-chat), using the official repository's code path.

    Mirrors eval_scripts/eval_ref.py from Vision-CAIR/MiniGPT-4: eval_configs/minigptv2_eval.yaml (8-bit LLM,
    448px BLIP-2 eval transform), the `[refer] give me the location of ...` task prompt inside the MiniGPT-v2 chat
    template, greedy decoding, and answers `{<x1><y1><x2><y2>}` on a 0-100 grid (scaled here to 0-1000).
    `low_resource=False` loads the LLM in fp16 instead of 8-bit (the repository's non-low-resource path).
    """

    def __init__(self, repo=MINIGPT4_REPO, llama_path=MINIGPTV2_LLAMA_PATH, ckpt=MINIGPTV2_CKPT, device=DEVICE, max_new_tokens=20,
                 low_resource=MINIGPTV2_LOW_RESOURCE):
        import argparse
        import sys
        sys.path.insert(0, os.path.abspath(repo))
        try:
            from minigpt4.common.config import Config
            from minigpt4.common.eval_utils import prepare_texts
            from minigpt4.common.registry import registry
            from minigpt4.conversation.conversation import CONV_VISION_minigptv2
            import minigpt4.models, minigpt4.processors  # noqa: F401  (register classes)
        except ImportError as e:
            raise ImportError(f"MiniGPT-v2 import failed ({e}); needs a clone of https://github.com/Vision-CAIR/MiniGPT-4 "
                              "(set VISTA_MINIGPT4_REPO) and the packages in env/minigptv2.txt") from e

        print(f"Loading MiniGPT-v2 (LLM {llama_path}, checkpoint {ckpt}, {'8-bit' if low_resource else 'fp16'} LLM)...")
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._prepare_texts = prepare_texts
        cfg = Config(argparse.Namespace(cfg_path=os.path.join(repo, "eval_configs/minigptv2_eval.yaml"),
                                        options=[f"model.llama_model={llama_path}", f"model.ckpt={ckpt}",
                                                 f"model.low_resource={low_resource}"]))
        self.model = registry.get_model_class(cfg.model_cfg.arch).from_config(cfg.model_cfg).to(device).eval()
        vis_cfg = cfg.datasets_cfg.get(list(cfg.datasets_cfg.keys())[0]).vis_processor.train
        self.vis_processor = registry.get_processor_class(vis_cfg.name).from_config(vis_cfg)
        self.conv_temp = CONV_VISION_minigptv2.copy()
        self.conv_temp.system = ""
        print("MiniGPT-v2 model loaded successfully!")

    def run_inference(self, image: Image.Image, question: str):
        """
        Runs inference on a single image and question.

        Returns:
            tuple: (text_output, boxes, query, response) to match the evaluation script's interface.
            boxes: list of tensors with [x1, y1, x2, y2] in 0-1000 scale
        """
        return self.run_inference_batch([image], question)[0]

    def run_inference_batch(self, images, question: str):
        """Same as run_inference for several frames that share one query, in one `generate` call (as the official
        eval_ref.py batches). All prompts have the same length, so no padding is involved."""
        prompt = f"[refer] give me the location of {question}"
        texts = self._prepare_texts([prompt] * len(images), self.conv_temp)
        batch = torch.stack([self.vis_processor(im.convert("RGB")) for im in images]).to(self.device)
        with torch.no_grad():
            responses = self.model.generate(batch, texts, max_new_tokens=self.max_new_tokens, do_sample=False)
        return [(r, self._parse_boxes(r), prompt, r) for r in responses]

    def _parse_boxes(self, text: str):
        """Extract `{<x1><y1><x2><y2>}` (0-100 grid) and scale to the 0-1000 grid."""
        boxes = []
        for match in re.findall(r'\{<(\d{1,3})><(\d{1,3})><(\d{1,3})><(\d{1,3})>\}', text):
            coords = [int(x) * 10 for x in match]
            boxes.append(torch.tensor(coords, device=self.device).unsqueeze(0))
        return boxes


# --------------------------
# REGISTRY
# --------------------------
# Models evaluated in the paper whose original inference code is not part of this repository
# (LLaVA-Grounding, Sphinx-v2) are listed in the README.
MODELS = {
    "cogvlm": CogVLMSingleSample,
    "shikra": ShikraSingleSample,
    "ferret": FerretSingleSample,
    "qwen3vl": Qwen3VLSingleSample,
    "mimovl": MiMoVLSingleSample,
    "internvl2.5": InternVL25SingleSample,
    "qwenvl": QwenVLChatSingleSample,
    "gdino": GroundingDINOSingleSample,
    "minigptv2": MiniGPTv2SingleSample,
}


def build_model(name):
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name]()
