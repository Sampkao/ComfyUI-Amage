import torch
import numpy as np
from torch import nn
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
from transformers import pipeline
import folder_paths

class AmageSTTNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),  # 前一節點的音訊
                "language": ("STRING", {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "transcribe"
    CATEGORY = "Amage/Audio"

    def __init__(self):
        model_path = folder_paths.models_dir + "/sonic/whisper-tiny"
        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=model_path,
            device=0 if torch.cuda.is_available() else -1
        )

    def transcribe(self, audio, language="auto"):
        try:
            # ---------- 1. 取值 ----------
            if audio is None:
                return ("",)

            # ComfyUI 的 AUDIO 通常是 dict { "waveform": tensor, "sample_rate": int }
            if isinstance(audio, dict) and "waveform" in audio:
                audio = audio["waveform"]

            # ---------- 2. 轉 numpy ----------
            if isinstance(audio, torch.Tensor):
                audio_np = audio.detach().cpu().numpy()
            elif isinstance(audio, np.ndarray):
                audio_np = audio
            else:
                return ("[STT Error] Unsupported audio type",)

            # ---------- 3. 砍掉 batch 維度（如果存在） ----------
            # 可能的 shape：
            #   (1, channels, samples)  -> (channels, samples)
            #   (1, samples, channels)  -> (samples, channels)
            if audio_np.ndim == 3 and audio_np.shape[0] == 1:
                audio_np = audio_np[0]

            # ---------- 4. 轉 float32 ----------
            audio_np = audio_np.astype(np.float32)

            # ---------- 5. 處理聲道 ----------
            if audio_np.ndim == 2:
                # channel-first (2, samples) 的情況
                if audio_np.shape[0] <= 2:
                    audio_np = np.mean(audio_np, axis=0)
                # channel-last (samples, 2) 的情況
                else:
                    audio_np = np.mean(audio_np, axis=1)

            if audio_np.ndim != 1:
                return (f"[STT Error] Unexpected final shape {audio_np.shape}",)

            # ---------- 6. 送入 Whisper ----------
            result = self.pipe(
                audio_np,
                generate_kwargs={"language": None if language == "auto" else language}
            )
            text = result.get("text", "").strip()
            return (text,)

        except Exception as e:
            return (f"[STT Error] {e}",)
        
class AmageFpsConverterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "original_fps": ("FLOAT", {"default": 25.0, "min": 0.1, "max": 120.0}),
                "target_fps":  ("FLOAT", {"default": 16.0, "min": 0.1, "max": 120.0}),
                "method": (["downsample", "duplicate", "interpolate"], {"default": "interpolate"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert_fps"
    CATEGORY = "Amage/Video"

    def convert_fps(self, images, original_fps, target_fps, method):

        original_fps = int(round(original_fps)) # 轉整數
        target_fps   = int(round(target_fps))
        
        # images 是 B×H×W×C 的 Tensor
        if original_fps == target_fps:
            return (images,)

        total_frames = images.shape[0]
        ratio = target_fps / original_fps
        new_tensors = []

        if method == "downsample":
            desired_frames = max(1, int(round(total_frames * ratio)))
            indices = np.linspace(0, total_frames - 1, desired_frames).astype(int)
            new_tensors = images[indices]

        elif method == "duplicate":
            repeat = max(1, int(round(ratio)))
            new_tensors = images.repeat_interleave(repeat, dim=0)

        elif method == "interpolate":
            # 僅在 ratio > 1 時插值
            if ratio <= 1.0 or total_frames < 2:
                new_tensors = images
            else:
                for i in range(total_frames - 1):
                    img1 = images[i].unsqueeze(0)
                    img2 = images[i + 1].unsqueeze(0)
                    new_tensors.append(img1)

                    num_interpolated = max(1, int(round(ratio - 1)))
                    for j in range(1, num_interpolated + 1):
                        alpha = j / (num_interpolated + 1)
                        interp = img1 * (1 - alpha) + img2 * alpha
                        new_tensors.append(interp)

                new_tensors.append(images[-1].unsqueeze(0))
                new_tensors = torch.cat(new_tensors, dim=0)

        return (new_tensors,)

class AmageOneNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": ("STRING", {"multiline": False, "default": ""}),
                "text":       ("STRING", {"multiline": True, "default": ""}),
                "width":      ("INT",    {"default": 1024}),
                "height":     ("INT",    {"default": 1024}),
                "aspect_ratio": ("STRING", {"default": "16:9"}),
                "denoise":      ("FLOAT",  {"default": 0.75, "step": 0.01}),
                "n_iter":     ("INT",    {"default": 1, "min": 1, "max": 100}),
                "string_1":   ("STRING", {"default": ""}),
                "int_1":      ("INT",    {"default": 1}),
                "float_1":    ("FLOAT",  {"default": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING","INT","INT","STRING","FLOAT","INT","STRING","INT","FLOAT")
    RETURN_NAMES = ("prompt_text", "width", "height", "aspect_ratio", "denoise", "n_iter", "string_1", "int_1", "float_1")
    FUNCTION = "amage_text"
    CATEGORY = "Text"

    def amage_text(
        self,
        input_text,
        text,
        width,
        height,
        aspect_ratio,
        denoise,
        n_iter,
        string_1,
        int_1,
        float_1,
    ):
        combined_text = f"{input_text} {text}".strip()
        return (combined_text, width, height, aspect_ratio, denoise, n_iter, string_1, int_1, float_1)
    
class AmageTextNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # "TEXT" 不是標準類型，建議使用 "STRING" 並設定 multiline
                "input_text": ("STRING", {"multiline": False, "default": ""}),
                "text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    # 返回類型也建議使用 "STRING"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "amage_text"
    CATEGORY = "Text" # 您可以建立子目錄，例如 "Amage/Text"

    def amage_text(self, input_text, text):
        # 組合後的文字，並移除頭尾多餘的空格
        combined_text = f"{input_text} {text}".strip()
        # 必須返回一個元組 (tuple)
        return (combined_text,)
    