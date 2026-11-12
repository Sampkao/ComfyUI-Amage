import torch
import numpy as np
import torchaudio.transforms as T
from torch import nn
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
from transformers import pipeline
import folder_paths
import gc

class AmageTextConvertNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "mode": ([
                    "t2s (繁轉簡)", 
                    "s2t (簡轉繁)", 
                    "zh-en (中翻英)", 
                    "en-zh (英翻中)"
                ], {"default": "t2s (繁轉簡)"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert_text"
    CATEGORY = "Amage/Text"

    def convert_text(self, text, mode):
        if not text or text.strip() == "":
            return ("",)
            
        try:
            # 處理本地繁簡轉換 (OpenCC)
            if "t2s" in mode or "s2t" in mode:
                from opencc import OpenCC
                # 提取模式代碼 (t2s 或 s2t)
                clean_mode = "t2s" if "t2s" in mode else "s2t"
                cc = OpenCC(clean_mode)
                return (cc.convert(text),)

            # 處理 Google 翻譯 (deep-translator)
            elif "zh-en" in mode or "en-zh" in mode:
                from deep_translator import GoogleTranslator

                if "zh-en" in mode:
                    # 中翻英：明確指定 source 為 zh-CN (相容繁簡)
                    translator = GoogleTranslator(source='zh-TW', target='en')
                    translated = translator.translate(text)
                else:
                    # 英翻中：目標改為 zh-CN
                    translator = GoogleTranslator(source='en', target='zh-TW')
                    translated = translator.translate(text)
                
                # 如果翻譯結果為空或出錯，回傳原文字並加上提示
                return (translated if translated else text,)

        except Exception as e:
            return (f"[Convert Error] {str(e)}",)
        
class AmageVramCleanerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any_input": ("*",), 
            },
            "optional": {
                "label": ("STRING", {"default": "Deep VRAM Cleanup"}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "clean_vram"
    CATEGORY = "Amage/Utils"

    def clean_vram(self, any_input, label=""):
        import comfy.model_management as model_management
        import gc
        import torch

        print(f"🔷 [Amage] {label} • Starting Deep Cleanup 🔷")

        try:
            # 1. 卸載所有 ComfyUI 追蹤的模型
            # 這會將模型移出顯存，是清理的最前端步驟
            model_management.unload_all_models()
            
            # 2. 同步硬體並執行第一波清理
            # 包含 torch.cuda.synchronize() 確保 GPU 任務已完成
            model_management.soft_empty_cache(True)

            # 3. 強制執行 Python 垃圾回收
            # 釋放所有不再使用的 Tensor 物件引用
            gc.collect()

            # 4. 深度 CUDA 清理 (參考範例)
            if torch.cuda.is_available():
                # 第一回清理：釋放剛被 GC 回收的顯存
                torch.cuda.empty_cache()
                
                # 清理 CUDA IPC (進程間通訊) 句柄，解決某些量化模型殘留顯存的問題
                torch.cuda.ipc_collect()
                
                # 掃描並移除死掉的模型包裝器 (Dead Wrappers)
                model_management.cleanup_models()
                
                # 第二回清理：確保殘留的碎片被徹底清空
                torch.cuda.empty_cache()
                
                # 重置峰值統計 (可選)
                try:
                    torch.cuda.reset_peak_memory_stats()
                except:
                    pass

            print(f"✅ [Amage] Deep Cleanup Complete.")

        except Exception as e:
            print(f"❌ [Amage] Cleanup Error: {e}")

        return (any_input,)
    
class AmageSTTNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",), 
                "language": ("STRING", {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "transcribe"
    CATEGORY = "Amage/Audio"

    def __init__(self):
        # 建議模型路徑保持一致
        model_path = folder_paths.models_dir + "/sonic/whisper-medium"
        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=model_path,
            chunk_length_s=30, # 修正長音訊報錯：自動切片
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 # 使用半精度可以跑更快，且依然節省一半顯存
        )

    def transcribe(self, audio, language="auto"):
        try:
            if audio is None:
                return ("",)

            # ---------- 1. 取值與採樣率校正 ----------
            waveform = audio["waveform"] # 假設格式為 [Batch, Channel, Samples]
            sr = audio["sample_rate"]

            # ---------- 2. 重採樣至 16000Hz (Whisper 必要條件) ----------
            if sr != 16000:
                resampler = T.Resample(sr, 16000)
                waveform = resampler(waveform)

            # ---------- 3. 轉為單聲道並標準化 ----------
            # 如果是多聲道 [B, C, S]，取平均值
            if waveform.ndim == 3:
                waveform = torch.mean(waveform, dim=1) 
            
            # 移除 Batch 維度，轉為 numpy float32
            audio_np = waveform.squeeze().cpu().numpy().astype(np.float32)

            # ---------- 4. 送入 Whisper (加入長格式參數) ----------
            # 設定 return_timestamps=True 解決超過 30 秒的報錯
            result = self.pipe(
                audio_np,
                generate_kwargs={
                    "language": None if language == "auto" else language,
                    "return_timestamps": True ,
                    "task": "transcribe"
                }
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
    
