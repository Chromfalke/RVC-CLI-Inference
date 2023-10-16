from scipy.io import wavfile
from fairseq import checkpoint_utils
from rvc.lib.audio import load_audio
from rvc.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from rvc.lib.vc_infer_pipeline import VC
from multiprocessing import cpu_count
import numpy as np
import torch
import sys


class Config:
    def __init__(self, device: str):
        self.device = device
        self.is_half = False
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()
        self.hubert_model = None
        self.cpt = None

    def device_config(self) -> tuple:
        if torch.cuda.is_available() and self.device != "cpu":
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                self.is_half = False
                for config_file in ["32k.json", "40k.json", "48k.json"]:
                    with open(f"configs/{config_file}", "r") as f:
                        strr = f.read().replace("true", "false")
                    with open(f"configs/{config_file}", "w") as f:
                        f.write(strr)
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            self.is_half = False

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max
    
    def load_hubert(self):
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            ["rvc/hubert_base.pt"],
            suffix="",
        )
        self.hubert_model = models[0]
        self.hubert_model = self.hubert_model.to(self.device)
        if self.is_half:
            self.hubert_model = self.hubert_model.half()
        else:
            self.hubert_model = self.hubert_model.float()
        self.hubert_model.eval()

    def get_vc(self, model_path: str):
        print("loading model %s" % model_path)
        self.cpt = torch.load(model_path, map_location="cpu")
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")
        if self.version == "v1":
            if self.if_f0 == 1:
                self.net_g = SynthesizerTrnMs256NSFsid(*self.cpt["config"], is_half=self.is_half)
            else:
                self.net_g = SynthesizerTrnMs256NSFsid_nono(*self.cpt["config"])
        elif self.version == "v2":
            if self.if_f0 == 1:
                self.net_g = SynthesizerTrnMs768NSFsid(*self.cpt["config"], is_half=self.is_half)
            else:
                self.net_g = SynthesizerTrnMs768NSFsid_nono(*self.cpt["config"])
        del self.net_g.enc_q
        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.device)
        if self.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()
        self.vc = VC(self.tgt_sr, self)

    def vc_single(self, input_audio_path: str, f0_up_key: int, f0_method: str, file_index: str, model_path: str, output_path: str):
        sid = 0
        index_rate = 1
        filter_radius = 3
        resample_sr = 0
        rms_mix_rate = 0
        protect = 0.33

        self.get_vc(model_path)
        if input_audio_path is None:
            print("You need to select an audio file")
            sys.exit(1)

        audio = load_audio(input_audio_path, 16000)
        audio_max = np.abs(audio).max() / 0.95

        if audio_max > 1:
            audio /= audio_max
        times = [0, 0, 0]

        if self.hubert_model == None:
            self.load_hubert()

        file_index = file_index.strip(" ").strip('"').strip("\n").strip('"').strip(" ").replace("trained", "added")


        audio_opt = self.vc.pipeline(
            self.hubert_model,
            self.net_g,
            sid,
            audio,
            input_audio_path,
            times,
            f0_up_key,
            f0_method,
            file_index,
            index_rate,
            self.if_f0,
            filter_radius,
            self.tgt_sr,
            resample_sr,
            rms_mix_rate,
            self.version,
            protect,
            None
        )
        wavfile.write(output_path, self.tgt_sr, audio_opt)
        print("processed")


def infer(transpose_value: int, input_file: str, output_folder: str, model_path: str, index_path: str, device: str, pitch_extraction_method: str):
    config = Config(device)
    config.load_hubert()
    config.vc_single(input_file, transpose_value, pitch_extraction_method, index_path, model_path, output_folder)
